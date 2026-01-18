# TruthfulQA Validation Report

Revision: TFQ
Models:   12
Questions: 169

## Correlation

  Metric       Value    p-value   Interpretation
  ----------   ------   -------   --------------
  Pearson r    0.8863    0.0001   strong
  Spearman     0.9231    0.0000   strong

## Score Comparison

  Rank  Model                      Peer   Truth  T.Rank
  ----  -------------------------  -----  -----  ------
     1  claude-opus-4-5             8.65   9.41       2
     2  claude-sonnet-4-5           8.59   9.76       1
     3  gemini-3-pro-preview        8.57   9.23       3
     4  gpt-5-mini                  8.41   8.58       6
     5  grok-4-1-fast               8.37   8.88       4
     6  gpt-5.2                     8.11   8.76       5
     7  gemini-3-flash-thinking     8.09   8.05       8
     8  deepseek-chat               8.03   7.63      10
     9  kimi-k2-0905                7.94   7.93       9
    10  sonar-pro                   7.52   8.11       7
    11  mistral-large               7.49   7.34      11
    12  llama-4-maverick            7.14   7.16      12

## Conclusion

Peer evaluation **strongly correlates** with ground truth (r=0.886).