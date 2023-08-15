import json
import pandas as pd
from sklearn.metrics import f1_score

from const import *

def _load_json():
    with open('data/exp_access/user.json') as f:
        _user = pd.DataFrame(json.load(f)).set_index('id')
    _user = _user[_user['age'] > 5]

    with open('data/exp_access/task_result.json') as f:
        _task_result = pd.DataFrame(json.load(f)).set_index('id')
    
    _instruction_quiz_answer = _task_result[_task_result['task_name'] == 'InstructionQuizAnswer'].copy()
    _main_task_result = _task_result[_task_result['task_name'] == 'MainTaskAnswer'].copy()
    _questionnaire = _task_result[_task_result['task_name'] == 'QuestionnaireAnswer'].copy()
    _open_question_answer = _task_result[_task_result['task_name'] == 'OpenQuestionAnswer'].copy()

    return _user, _instruction_quiz_answer, _main_task_result, _questionnaire, _open_question_answer

def load():
    _user, _instruction_quiz_answer, _main_task_result, _questionnaire, _open_question_answer = _load_json()

    def answer_to_str(grp):
        array = grp.array
        return tuple(line['value'] for line in array)

    _instruction_quiz_correct = (_instruction_quiz_answer.sort_values(('item_name')).groupby('user_id')['answers'].apply(answer_to_str) == (2,2,1,1)).index

    _main_task_result_details = pd.DataFrame(list(_main_task_result['answers'].values), index=_main_task_result['answers'].index)
    _main_task_result_details['ai_correct'] = _main_task_result_details['ground_truth'] == _main_task_result_details['ai_answer']
    _main_task_result_details['final_answer_correct'] = _main_task_result_details['ground_truth'] == _main_task_result_details['final_answer'] 
    _main_task_result_details['cue_shown'] = _main_task_result_details['system_action'] != 'neutral'
    _main_task_result_details['system_action_token'] = _main_task_result_details['system_action'].map(CUE_TABLE)
    _main_task_result_details['user_decision_token'] = _main_task_result_details['user_decision'].map(DECISION_TABLE)
    _main_task_result_details = _main_task_result_details.drop(columns=[col for col in _main_task_result_details.columns if col in _main_task_result.columns])
    _main_task_result = pd.concat([
        _main_task_result,
        _main_task_result_details
        ], axis=1).drop_duplicates(['user_id', 'episode_id'])

    _num_trial_per_user = _main_task_result.groupby('user_id').apply(len)
    _completed_user = (_num_trial_per_user.groupby('user_id').apply(len) == 60).index


    valid_user = _instruction_quiz_correct.intersection(_completed_user).intersection(_user[_user['age'] != 0].index).intersection(_user[_user['is_no_more_login'] == True].index)
    task_result = _main_task_result[_main_task_result['user_id'].isin(valid_user)]


    f_score = task_result.groupby('user_id').apply(
        lambda grp:
            pd.Series(dict(
                f_score=f1_score(grp['ai_correct'], grp['user_decision'] == 'AI'),
                accuracy=grp['final_answer_correct'].mean(),
                condition=grp['user_class'].iloc[0],
                rate=grp['cue_shown'].mean(),
                reliance=( grp['user_decision'] == 'AI').mean()
                )
            )
    )

    _questionnaire['answers'] = _questionnaire['answers'].map(lambda x: x['value'])
    _questionnaire = _questionnaire.pivot_table(values='answers', index='user_id', columns='item_name')
    questionnaire = _questionnaire[_questionnaire.index.isin(valid_user)]

    _open_question_answer = _open_question_answer[_open_question_answer['user_id'].isin(valid_user)]
    _open_question_answer['answers'] = _open_question_answer['answers'].map(lambda x: x['value'])
    open_question_answer = _open_question_answer

    return valid_user, task_result, f_score, questionnaire, open_question_answer

