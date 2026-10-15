"""Restricted answer extraction/equivalence for math generations.

The parser accepts a deliberately small arithmetic language and never calls
``eval``, SymPy ``parse_expr``, or a language interpreter on model text. Values
are exact Fractions. Unsupported symbolic answers use normalized literal equality
and are reported as unsupported rather than guessed equivalent. Callers should
run this module in a subprocess with a wall-clock timeout for untrusted output.
"""
from __future__ import annotations
import ast
from fractions import Fraction
import math
import re
from dataclasses import dataclass


MAX_CHARS=4096; MAX_NODES=256; MAX_DENOMINATOR=10**12
_BOXED=re.compile(r"\\boxed\s*(?:\{([^{}]*)\}|\(([^()]*)\))",re.S)
_DOLLAR=re.compile(r"\$([^$]*)\$",re.S)
_NUMBER=re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class ParseRejected(ValueError): pass


@dataclass(frozen=True)
class Parsed:
    raw: str
    value: Fraction|None
    canonical: str
    supported: bool
    reason: str=''


def _balanced_box(text):
    """Extract the final boxed body with nested braces."""
    starts=[m.start() for m in re.finditer(r"\\boxed\s*\{",text)]
    if not starts:return None
    start=starts[-1]; i=text.find('{',start); depth=0
    for j in range(i,len(text)):
        if text[j]=='{' and (j==0 or text[j-1]!='\\'):depth+=1
        elif text[j]=='}' and (j==0 or text[j-1]!='\\'):
            depth-=1
            if depth==0:return text[i+1:j]
    return None


def extract_candidate(text, dataset='gsm8k'):
    if not isinstance(text,str) or len(text)>MAX_CHARS:raise ParseRejected('text too long')
    body=_balanced_box(text)
    if body is None:
        matches=_DOLLAR.findall(text)
        body=matches[-1] if matches else (text.split('####',1)[-1] if '####' in text else text)
    body=body.strip()
    if dataset.lower().startswith('gsm') and '####' in text:
        tail=text.rsplit('####',1)[1].strip()
        if tail:body=tail
    return body[:MAX_CHARS]


def _frac_braces(text):
    # Repeatedly replace numeric/nested numeric TeX fractions. No arbitrary
    # TeX commands survive into the arithmetic grammar.
    text=text.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    while '\\frac' in text:
        pos=text.rfind('\\frac'); i=pos+5
        while i<len(text) and text[i].isspace():i+=1
        if i>=len(text) or text[i]!='{':raise ParseRejected('malformed fraction')
        def group(k):
            if k>=len(text) or text[k]!='{':raise ParseRejected('malformed group')
            d=1;j=k+1
            while j<len(text):
                if text[j]=='{' and text[j-1]!='\\':d+=1
                elif text[j]=='}' and text[j-1]!='\\':
                    d-=1
                    if not d:return text[k+1:j],j+1
                j+=1
            raise ParseRejected('unclosed group')
        a,j=group(i); b,j=group(j); text=text[:pos]+'('+a+')/('+b+')'+text[j:]
    return text


def _prepare(body):
    text=_frac_braces(body)
    text=text.replace('\\left','').replace('\\right','').replace('\\!','')
    text=text.replace('−','-').replace('–','-').replace('×','*').replace('÷','/')
    text=text.replace('\\cdot','*').replace('\\times','*').replace('^','**')
    text=re.sub(r"\\sqrt\s*\{([^{}]*)\}",r"sqrt(\1)",text)
    text=re.sub(r"\\text\s*\{[^{}]*\}","",text)
    text=text.replace('{','(').replace('}',')')
    text=text.replace('\\pi','pi')
    text=re.sub(r"(?<=\d),(?=\d)","",text)
    text=re.sub(r"\s+","",text)
    # Units and prose are not arithmetic.
    text=re.sub(r"(?:dollars?|cm|m|kg|hours?|minutes?)$","",text,flags=re.I)
    return text


def _safe_fraction(expr):
    if len(expr)>512:raise ParseRejected('expression too long')
    tree=ast.parse(expr,mode='eval'); count=0
    def visit(node):
        nonlocal count
        count+=1
        if count>MAX_NODES:raise ParseRejected('too many nodes')
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and not isinstance(node.value,bool):
            if isinstance(node.value,float) and (not math.isfinite(node.value)):raise ParseRejected('nonfinite')
            return Fraction(str(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v=visit(node.operand);return v if isinstance(node.op,ast.UAdd) else -v
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a=visit(node.left);b=visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):
                if not b:return (_ for _ in ()).throw(ParseRejected('division by zero'))
                return a/b
            if b.denominator!=1 or abs(b.numerator)>12:return (_ for _ in ()).throw(ParseRejected('unsafe exponent'))
            return a**b.numerator
        if isinstance(node,ast.Name) and node.id=='pi':return None
        raise ParseRejected('unsupported syntax')
    value=visit(tree)
    if value is None:raise ParseRejected('symbolic constant')
    if abs(value.denominator)>MAX_DENOMINATOR:raise ParseRejected('denominator too large')
    return value


def parse_candidate(text,dataset='gsm8k'):
    raw=extract_candidate(text,dataset); prepared=_prepare(raw)
    try:value=_safe_fraction(prepared)
    except (SyntaxError,ValueError,ParseRejected,ZeroDivisionError) as exc:
        canonical=re.sub(r"\\(left|right|,|!|;|,)","",raw.lower())
        canonical=re.sub(r"\s+","",canonical)
        return Parsed(raw,None,canonical,False,str(exc))
    return Parsed(raw,value,str(value),True,'')


def equivalent(prediction,reference,dataset='gsm8k'):
    a=parse_candidate(prediction,dataset); b=parse_candidate(reference,dataset)
    if a.supported and b.supported:return a.value==b.value,{'mode':'exact_fraction','prediction':a.canonical,'reference':b.canonical}
    if not a.supported or not b.supported:return False,{'mode':'unsupported','prediction_reason':a.reason,'reference_reason':b.reason}
    return a.canonical==b.canonical,{'mode':'literal'}
