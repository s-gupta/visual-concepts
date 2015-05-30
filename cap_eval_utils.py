import numpy as np
import sg_utils as utils
from scipy.interpolate import interp1d

from IPython.core.debugger import Tracer

def calc_pr_ovr(counts, out, K):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
    K      : number of references
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """
  K = np.float64(K)
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts*(1.-1./K);
  fp = sortcounts.copy();
  for i in xrange(sortcounts.shape[0]):
    if sortcounts[i] > 1:
      fp[i] = 0.;
    elif sortcounts[i] == 0:
      fp[i] = 1.;
    elif sortcounts[i] == 1:
      fp[i] = 1./K;
  
  P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));

  # c = accumarray(sortcounts(:)+1, 1);
  c = [np.sum(np.array(sortcounts) == i) for i in xrange(int(max(sortcounts)+1))]
  ind = np.array(range(0, len(c)));
  numinst = ind*c*(K-1.)/K;
  numinst = np.sum(numinst, axis = 0)
  R = np.cumsum(tp)/numinst
  
  ap = voc_ap(R,P)
  return P, R, score, ap


def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Computes the AP under the precision recall curve.
  """

  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def compute_precision_score_mapping(thresh, prec, score):
  ind = np.argsort(thresh);
  thresh = thresh[ind];
  prec = prec[ind];
  for i in xrange(1, len(prec)):
    prec[i] = max(prec[i], prec[i-1]);
  
  indexes = np.unique(thresh, return_index=True)[1]
  indexes = np.sort(indexes);
  thresh = thresh[indexes]
  prec = prec[indexes]
  
  thresh = np.vstack((min(-1000, min(thresh)-1), thresh[:, np.newaxis], max(1000, max(thresh)+1)));
  prec = np.vstack((prec[0], prec[:, np.newaxis], prec[-1]));
  
  f = interp1d(thresh[:,0], prec[:,0])
  val = f(score)
  return val

def human_agreement(gt, K):
  """
  function [prec, recall] = human_agreement(gt, K)
  """
  c = np.zeros((K+1,1), dtype=np.float64)
  for i in xrange(len(gt)):
    c[gt[i]] += 1;
  
  c = c/np.sum(c);
  ind = np.array(range(len(c)))[:, np.newaxis]

  n_tp = sum(ind*(ind-1)*c)/K;
  n_fp = c[1]/K;
  numinst = np.sum(c * (K-1) * ind) / K;
  prec = n_tp / (n_tp+n_fp);
  recall = n_tp / numinst;
  
  return prec, recall
