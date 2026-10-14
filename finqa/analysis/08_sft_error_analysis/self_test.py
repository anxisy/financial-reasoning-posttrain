# -*- coding: utf-8 -*-
from error_analysis import classify

def main():
    pred={'prediction':'Step 1: divide(60,243)=25%\\n<answer>25%</answer>','prediction_answer':'25%','gold_answer':'24.69%','metadata':{'operators':['divide','multiply'],'num_steps':2,'annotation_quality':'high','semantic_status':'not_flagged'}}
    out=classify(pred,{'prompt':[{'role':'user','content':'Canada 60. Total 243. What percent?'}]})
    assert 'near_numeric_miss' in out['categories'] and 'multi_step_failure' in out['categories']
    pred2={'prediction':'<answer>0.5</answer>','prediction_answer':'0.5','gold_answer':'50%','metadata':{'operators':['divide'],'num_steps':1,'annotation_quality':'high','semantic_status':'not_flagged'}}
    out2=classify(pred2,{'prompt':[]}); assert 'percentage_scale_error' in out2['categories']
    pred3={'prediction':'<answer>no</answer>','prediction_answer':'no','gold_answer':'yes','metadata':{'operators':['greater'],'num_steps':1,'annotation_quality':'high','semantic_status':'not_flagged'}}
    out3=classify(pred3,{'prompt':[]}); assert 'yes_no_error' in out3['categories']
    print('SELF TEST PASSED')
if __name__=='__main__':main()
