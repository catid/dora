import json
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from rankstudy.v3.answer_parser import equivalent,parse_candidate,ParseRejected


@pytest.mark.parametrize('prediction,reference',[
 ('#### 42','42'),('\\boxed{\\frac{14}{3}}','14/3'),
 ('The answer is \\boxed{3}','3'),('\\boxed{( 3, \\frac{\\pi}{2} )}','(3,pi/2)'),
])
def test_restricted_parser_cases(prediction,reference):
    ok,detail=equivalent(prediction,reference)
    if 'pi' in prediction:assert not ok and detail['mode']=='unsupported'
    else:assert ok and detail['mode']=='exact_fraction'


def test_no_python_calls_or_names_are_accepted():
    for text in ['__import__("os").system("touch /tmp/bad")','1+foo','open("x")','1//2','1<<2']:
        p=parse_candidate(text);assert not p.supported


def test_public_gsm_train_answers_have_extractable_numeric_subset():
    table=pq.read_table('/var/tmp/dora-adaptive-rank/data/openai_gsm8k/main/train-00000-of-00001.parquet').to_pylist()
    checked=0;supported=0
    for row in table[:512]:
        answer=row['answer']; final=answer.rsplit('####',1)[-1].strip();checked+=1
        if parse_candidate(final).supported:supported+=1
    assert checked==512 and supported>=500


def test_bounds_reject_deep_or_large_inputs():
    with pytest.raises(ParseRejected):parse_candidate('1'*5000)
    assert not parse_candidate('+'.join(['1']*300)).supported
