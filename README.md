
We annotate experiencer information for benchmark datasets (Xia and Ding, 2019). The updated datasets named as “all_data_pair(experiencers)”, and there are json versions and txt versions. The JSON format is structural representation, in which contains the experiencers and the retrieved common knowledge (Xreact and Xwant) from ATMOIC corresponding to each clause. Readers can directly implement expresser-specific and knowledge-specific emotion cause extraction based on the corpus.

**File introduction**

1. "**coreference_data**"--This catalog provides the results of anaphora resolution using the Stanford University anaphora Resolution Tool (https://stanfordnlp.github.io/CoreNLP/)

2. "**commonsense_data**"--In this catalog, we put into the retrieved commonsense knowledge for each clause of ECPE datasets and store in ecpe_data_commonsense.pkl. Note that, the file 'ATOMIC_Chinese.tsv' is lacked, we need to download at link (https://github.com/XiaoMi/C3KG)

3. "**graph_build.py**"--This file is used to build experiencer-driven graph and knowledge-aware graph. The process of graph construction needs coreference data and syntactic analysis results using the tool (http://ospm9rsnd.bkt.clouddn.com/model/ltp_data_v3.4.0.zip)

4. "**knowledge_extractor.py**"-- This file is used to retrieve commonsense knowledges from ATOMIC.
