"""Training-only pair features for a pending generic binary text matcher."""
from collections import Counter,defaultdict
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def normalize_pairs(pairs):
    if not isinstance(pairs,list) or not pairs:raise ValueError('Expected nonempty text-pair list')
    result=[]
    for pair in pairs:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('Expected two text fields')
        normalized=[]
        for text in pair:
            if type(text) is not str or len(text)>512:raise ValueError('Expected text up to 512 characters')
            text=' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))
            if not text:raise ValueError('Text is empty after normalization')
            normalized.append(text)
        result.append(tuple(normalized))
    return result


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def _jaccard(a,b):return len(a&b)/len(a|b) if a|b else 1.


class PairFeatures:
    """Lexical, LSA and unlabeled graph features fitted exclusively on training.

Training transformations exclude the current pair's single graph occurrence.
Query transformations never update vocabulary, embeddings or graph counts.
Text symmetry is preserved. Context denotes candidate-pair co-occurrence, not
verified semantic equivalence; labels are never supplied to this component.
"""
    def fit(self,pairs,*,components=2,seed=42):
        if type(components) is not int or components<1 or type(seed) is not int or not 0<=seed<2**32:
            raise ValueError('Invalid LSA controls')
        normalized=normalize_pairs(pairs)
        docs=sorted({text for pair in normalized for text in pair})
        vectorizer=TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',lowercase=False)
        matrix=vectorizer.fit_transform(docs)
        if components>min(matrix.shape)-1:raise ValueError('Insufficient training vocabulary/documents for requested LSA dimension')
        svd=TruncatedSVD(n_components=components,random_state=seed)
        svd.fit(matrix)
        self.vectorizer,self.svd=vectorizer,svd
        self.pairs=Counter(tuple(sorted(pair)) for pair in normalized)
        self.adjacency=defaultdict(Counter)
        for a,b in normalized:
            self.adjacency[a][b]+=1
            if a!=b:self.adjacency[b][a]+=1
        self.fitted_pairs=normalized
        return self

    def _features(self,normalized,exclude_self):
        if not hasattr(self,'vectorizer'):raise ValueError('Feature extractor is not fitted')
        left=self.vectorizer.transform([p[0] for p in normalized]);right=self.vectorizer.transform([p[1] for p in normalized])
        lhs=self.svd.transform(left);rhs=self.svd.transform(right)
        tfidf=np.asarray(left.multiply(right).sum(axis=1)).ravel()
        result=[]
        for i,(a,b) in enumerate(normalized):
            ta,tb=set(a.split()),set(b.split())
            ca={a[j:j+3] for j in range(max(1,len(a)-2))};cb={b[j:j+3] for j in range(max(1,len(b)-2))}
            na=self.adjacency.get(a,Counter()).copy();nb=self.adjacency.get(b,Counter()).copy()
            count=self.pairs[tuple(sorted((a,b)))]
            if exclude_self:
                na[b]-=1;nb[a]-=1;count-=1
            neighbors_a={n for n,c in na.items() if c>0};neighbors_b={n for n,c in nb.items() if c>0}
            degree_a=sum(max(0,c) for c in na.values());degree_b=sum(max(0,c) for c in nb.values())
            norm=np.linalg.norm(lhs[i])*np.linalg.norm(rhs[i])
            cosine=float(lhs[i]@rhs[i]/norm) if norm else 0.
            result.append([float(a==b),_jaccard(ta,tb),_jaccard(ca,cb),min(len(a),len(b))/max(len(a),len(b)),
                1-edit_distance(a,b)/max(len(a),len(b)),tfidf[i],cosine,float(np.linalg.norm(lhs[i]-rhs[i])),
                min(degree_a,degree_b),max(degree_a,degree_b),_jaccard(neighbors_a,neighbors_b),count])
        values=np.asarray(result,dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite pair features')
        return values

    def transform_training(self):
        return self._features(self.fitted_pairs,True)

    def transform(self,pairs):
        return self._features(normalize_pairs(pairs),False)
