The evaluation scripts are real-time-filtering-modelA-eval.py and
real-time-filtering-modelB-eval.py for tasks A and B respectively.
Each script takes three arguments, the judgments file, the clusters file,
and the run file. So, to invoke the Task A script on run 'runA', for example,
use:
    python real-time-filtering-modelA-eval.py -q qrels.txt -c clusters-2015.json -r runA

The judgment file is qrels.txt.  Judgment sets were built as described below.
All runs contributed to the pools.
    * For task A runs, for each topic, the first 10 (by delivery decision
      time) tweets per day are added to the pool.
    * For task B runs, for each topic, up to 85 (the pool depth) tweets are
      added to the pool, doing a round-robin by rank across days.  That is,
      first all rank 1 tweets from all days with tweets are added, then all
      rank 2 tweets, etc.  If one day runs out of tweets before the limit is
      reached, more tweets from those days that still have tweets are added
      until either there are no more tweets to add or the 85-tweet limit
      is reached.

NIST assessors judged these pools for 51 topics, assigning relevance labels
of 0 for not relevant, 1 for relevant, and 2 for highly relevant.  The
51 topics judged are:
	226,227,228,236,242,243,246,248,249,253,
	254,255,260,262,265,267,278,284,287,298,
	305,324,326,331,339,344,348,353,354,357,
	359,362,366,371,377,379,383,384,389,391,
	392,401,400,405,409,416,419,432,434,439,448

Tweets were then manually clustered into equivalence classes.  An unjudged
retweet encountered in a run was mapped to its corresponding cluster and
assigned a relevance judgment based on the label of the majority
of the judged tweets in that cluster.  These 'propagated' judgments were
added to the qrels file using different labels to distinguish them
from assessor-judged tweets.  The labels for propagated judgments are
-1 for not relevant, 3 for relevant and 4 for highly relevant.

The clusters file is clusters-2015.json.  As noted above, this file defines
the manually-produced equivalence classes of tweets. 
