use crate::base::*;
use crate::filters::*;
use pest;
use pest::Parser;
use std::collections::BTreeMap;
use std::sync::mpsc::channel;
use std::sync::Arc;
use threadpool::ThreadPool;

#[derive(Parser)]
#[grammar = "filter_expr.pest"]
struct FilterExprParser;

#[derive(PartialEq, Debug)]
enum FilterExpr {
    Channel(String),
    Filter(String, Box<FilterExpr>),
    Gain(Gain, Box<FilterExpr>),
    Mix(Vec<FilterExpr>),
}

fn parse_filter_expr(expr_str: &str) -> Result<FilterExpr> {
    use pest::iterators::Pair;
    fn parse_expr(pair: Pair<Rule>) -> FilterExpr {
        match pair.as_rule() {
            Rule::channel => FilterExpr::Channel(
                pair.into_inner()
                    .next()
                    .unwrap()
                    .as_span()
                    .as_str()
                    .to_string(),
            ),
            Rule::filter => {
                let mut inner_rules = pair.into_inner();
                FilterExpr::Filter(
                    inner_rules.next().unwrap().as_span().as_str().to_string(),
                    Box::new(parse_expr(inner_rules.next().unwrap())),
                )
            }
            Rule::mult_operand => parse_expr(pair.into_inner().next().unwrap()),
            Rule::mult => {
                let mut inner_rules = pair.into_inner();
                let db = inner_rules
                    .next()
                    .unwrap()
                    .into_inner()
                    .next()
                    .unwrap()
                    .as_span()
                    .as_str()
                    .parse::<f32>()
                    .unwrap();
                FilterExpr::Gain(
                    Gain { db },
                    Box::new(parse_expr(inner_rules.next().unwrap())),
                )
            }
            Rule::sum_operand => parse_expr(pair.into_inner().next().unwrap()),
            Rule::sum => FilterExpr::Mix(pair.into_inner().map(parse_expr).collect()),
            Rule::paren_expr => parse_expr(pair.into_inner().next().unwrap()),
            Rule::expression => parse_expr(pair.into_inner().next().unwrap()),

            Rule::WHITESPACE
            | Rule::letter
            | Rule::name_char
            | Rule::name
            | Rule::number
            | Rule::digit
            | Rule::level => unreachable!(),
        }
    }

    let expr = FilterExprParser::parse(Rule::expression, expr_str)?
        .next()
        .unwrap();
    Ok(parse_expr(expr))
}

trait ChannelSource: Send {
    fn apply(&mut self, frame: &Frame, output: &mut Vec<f32>);
    fn reset(&mut self);
}

struct DirectChannelSource {
    channel: ChannelPos,
}

impl ChannelSource for DirectChannelSource {
    fn apply(&mut self, frame: &Frame, output: &mut Vec<f32>) {
        *output = match frame.get_channel(self.channel) {
            Some(samples) => samples.to_vec(),
            None => vec![0.0; frame.len()],
        }
    }
    fn reset(&mut self) {}
}

struct FilterChannelSource {
    filter: Box<dyn AudioFilter>,
    base: Box<dyn ChannelSource>,
}

impl ChannelSource for FilterChannelSource {
    fn apply(&mut self, frame: &Frame, output: &mut Vec<f32>) {
        self.base.apply(frame, output);
        self.filter.apply_multi(output);
    }
    fn reset(&mut self) {
        self.filter.reset();
        self.base.reset();
    }
}

struct GainChannelSource {
    mult: f32,
    base: Box<dyn ChannelSource>,
}

impl ChannelSource for GainChannelSource {
    fn apply(&mut self, frame: &Frame, output: &mut Vec<f32>) {
        self.base.apply(frame, output);
        for i in 0..output.len() {
            output[i] *= self.mult;
        }
    }
    fn reset(&mut self) {
        self.base.reset();
    }
}

struct MixChannelSource {
    terms: Vec<Box<dyn ChannelSource>>,
}

impl ChannelSource for MixChannelSource {
    fn apply(&mut self, frame: &Frame, output: &mut Vec<f32>) {
        self.terms[0].apply(frame, output);
        let mut term_out = vec![];
        for term in self.terms[1..].iter_mut() {
            term.apply(frame, &mut term_out);
            for p in 0..output.len() {
                output[p] += term_out[p];
            }
        }
    }
    fn reset(&mut self) {
        for t in self.terms.iter_mut() {
            t.reset()
        }
    }
}

enum FilterConfig {
    Biquad(MultiBiquadParams),
    Fir(FirFilterParams),
}

fn filter_expr_to_channel_source(
    filter_expr: &FilterExpr,
    filters: &BTreeMap<String, FilterConfig>,
) -> Result<Box<dyn ChannelSource>> {
    let result: Box<dyn ChannelSource> = match filter_expr {
        FilterExpr::Channel(channel_id) => Box::new(DirectChannelSource {
            channel: parse_channel_id(channel_id)
                .ok_or_else(|| Error::from_string(format!("Unknown channel: {}", channel_id)))?,
        }),
        FilterExpr::Filter(filter_id, expr) => {
            let filter_config = filters
                .get(filter_id)
                .ok_or_else(|| Error::from_string(format!("Unknown filter: {}", filter_id)))?;
            let filter: Box<dyn AudioFilter> = match filter_config {
                FilterConfig::Biquad(biquad_params) => {
                    Box::new(MultiBiquadFilter::new(biquad_params))
                }
                FilterConfig::Fir(fir_params) => Box::new(FirFilter::new(fir_params)),
            };
            Box::new(FilterChannelSource {
                filter: filter,
                base: filter_expr_to_channel_source(expr, filters)?,
            })
        }
        FilterExpr::Gain(gain, expr) => Box::new(GainChannelSource {
            mult: gain.get_multiplier(),
            base: filter_expr_to_channel_source(expr, filters)?,
        }),
        FilterExpr::Mix(exprs) => {
            let mut terms = Vec::new();
            for e in exprs {
                terms.push(filter_expr_to_channel_source(e, filters)?)
            }
            Box::new(MixChannelSource { terms })
        }
    };
    Ok(result)
}

struct FilterExprProcessor {
    names: BTreeMap<String, ChannelPos>,
    channels: Vec<(ChannelPos, Box<dyn ChannelSource>)>,
    threadpool: ThreadPool,
}

impl FilterExprProcessor {
    fn new(
        filters: BTreeMap<String, FilterConfig>,
        channel_exprs: Vec<(String, FilterExpr)>,
    ) -> Result<FilterExprProcessor> {
        let mut names = BTreeMap::new();
        let mut channels = Vec::new();
        let mut next_pos = CHANNEL_DYNAMIC_BASE;
        for (name, expr) in channel_exprs {
            let pos = next_pos;
            next_pos += 1;
            if parse_channel_id(&name).is_some() || names.get(&name).is_some() {
                return Err(Error::from_string(format!(
                    "Filtered channel name is not unique: {}",
                    name
                )));
            }
            names.insert(name, pos);
            let source = filter_expr_to_channel_source(&expr, &filters)?;
            channels.push((pos, source));
        }

        let threadpool = ThreadPool::new(channels.len());
        Ok(FilterExprProcessor {
            names,
            channels,
            threadpool,
        })
    }

    fn get_channel_pos(&self, name: &str) -> Option<ChannelPos> {
        let result = parse_channel_id(name);
        if result.is_some() {
            return result;
        }
        self.names.get(name).map(|x| *x)
    }
}

impl StreamFilter for FilterExprProcessor {
    fn apply(&mut self, frame: Frame) -> Frame {
        let frame_arc = Arc::new(frame);

        let (tx, rx) = channel();
        for (channel, mut source) in self.channels.drain(..) {
            let tx = tx.clone();
            let frame_arc = frame_arc.clone();
            self.threadpool.execute(move || {
                let mut result = Vec::new();
                source.apply(&*frame_arc, &mut result);
                tx.send((channel, source, result));
            });
        }
        let mut results = Vec::new();
        for (channel, source, result) in rx.iter() {
            results.push((channel, result));
            self.channels.push((channel, source));
        }
        let mut frame = match Arc::try_unwrap(frame_arc) {
            Ok(f) => f,
            Err(_) => panic!(),
        };
        for (channel, pcm) in results {
            frame.set_channel(channel, pcm);
        }
        frame
    }

    fn reset(&mut self) {
        for (_, f) in self.channels.iter_mut() {
            f.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        assert!(parse_filter_expr("Left").unwrap() == FilterExpr::Channel("Left".to_string()));
        assert!(
            parse_filter_expr("f_1(A)").unwrap()
                == FilterExpr::Filter(
                    "f_1".to_string(),
                    Box::new(FilterExpr::Channel("A".to_string()))
                )
        );
        assert!(
            parse_filter_expr("Sub_rc(LFE + -10db * (LP_80(L + R) + LP_100(C)))").unwrap()
                == FilterExpr::Filter(
                    "Sub_rc".to_string(),
                    Box::new(FilterExpr::Mix(vec![
                        FilterExpr::Channel("LFE".to_string()),
                        FilterExpr::Gain(
                            Gain { db: -10.0 },
                            Box::new(FilterExpr::Mix(vec![
                                FilterExpr::Filter(
                                    "LP_80".to_string(),
                                    Box::new(FilterExpr::Mix(vec![
                                        FilterExpr::Channel("L".to_string()),
                                        FilterExpr::Channel("R".to_string())
                                    ]))
                                ),
                                FilterExpr::Filter(
                                    "LP_100".to_string(),
                                    Box::new(FilterExpr::Channel("C".to_string()))
                                ),
                            ])),
                        ),
                    ]))
                )
        );
    }
}
