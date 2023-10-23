This file contains an project.ipynb with the source code of bi-encoder, cross-encoder, svm and lstm models and the evaluation results for the project.
The ipynb file contains the following functions:

Function name(parameters)
1. prepocess(dataset, test) - for prepocess the original dataset
2. unmatched_data(claims_data, evid_data, evid_id, evid_num, batch_size=300000, k_=10, l=5, test=False)
- for sort out useful data for cross-encoder fine-tuning
 '''
  parameter description:
  claims_data: embedded claim sentences 
  evid_data: embedded evidence sentences 
  evid_id: evidence ID
  evid_num: evidence sentences - for measuring the number of evidences
  batch_size: number to determine batch size for scoring, preventing memory run out
  k: number of top-k unmatched evidence will be output for fine-tuning
  l: number of top-n to used for cross-encoder to predict related evidence
 '''
3. mnr_dataloader(claim_data, evid_data) - produce suitable dataset for MultipleNegativesRankingLoss training
4. contrastive_dataloader(claim_data, evid_data, unmatched_data, all_claim, batch_size=16)
   - produce suitable dataset for OnlineContrastiveLoss training, batch_size to control training batch and prevent memory issue
5. cross_prediction(claim_data, top_evids) - corss-encoder prediction for related evidence
6.  evid_results(top_evids, evid_scores, predictions, thres, test=False) - output the predicted related evidence
7. evid_combined(evid_id) - merge related evidence, prepocess for classification
8. convert_label(data, num) - convert claim label to number/words
9. output_dic(claim_id, evid_result, label_result, claim) - combine prediced data and generate a dictionary 