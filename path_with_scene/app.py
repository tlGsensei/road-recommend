import json
from flask import Flask, Response
from flask import request
from path_model import *


app = Flask(__name__)
'''
def class3_path(user_id, seq, target):
    saved_3_path = './savedir/class3_seq.pth'
    class3_shixun_dict = {'eh5oxkm9':1,'a2ct98o7':2,'klp26sqc':3,'vnw2fg5r':4,'8g93nfvc':5,'r4vlju5x':6,'guzqi4nm':7,'85fat9w3':8,'obtfwj3e':9}
    reverse_dict = {v: k for k, v in class3_shixun_dict.items()}
    seq_id = [class3_shixun_dict[x] for x in seq]
    target_id = class3_shixun_dict[target]
    path_id = load(saved_3_path, int(user_id), seq_id, target_id)
    path = [reverse_dict[x] for x in path_id]
    return path
'''

def class3_path(user_id, scene, seq, target):
    saved_3_path = './savedir/class3_seq.pth'
    class3_shixun_dict = {'eh5oxkm9':1,'a2ct98o7':2,'klp26sqc':3,'vnw2fg5r':4,'8g93nfvc':5,'r4vlju5x':6,'guzqi4nm':7,'85fat9w3':8,'obtfwj3e':9}
    reverse_dict = {v: k for k, v in class3_shixun_dict.items()}
    seq_id = [class3_shixun_dict[x] for x in seq]
    target_id = class3_shixun_dict[target]
    path_id = load_with_scene(saved_3_path, scene, int(user_id), seq_id, target_id)
    path = [reverse_dict[x] for x in path_id]
    return path

def class4_path(user_id, scene, seq, target):
    saved_4_path = './savedir/class4_seq.pth'
    class4_shixun_dict = {'zlg2nmcf':1,'3ozvy5f8':2,'b6ljcet3':3,'mbgfitn6':4,'uc64f2qs':5,'vtnag4op':6,'w3vcokrg':7,'uywljq4v':8,'ba56rk8v':9}
    reverse_dict = {v: k for k, v in class4_shixun_dict.items()}
    seq_id = [class4_shixun_dict[x] for x in seq]
    target_id = class4_shixun_dict[target]
    path_id = load_with_scene(saved_4_path, scene, int(user_id), seq_id, target_id)
    path = [reverse_dict[x] for x in path_id]
    return path
'''
def class4_path(user_id, seq, target):
    saved_4_path = './savedir/class4_seq.pth'
    class4_shixun_dict = {'zlg2nmcf':1,'3ozvy5f8':2,'b6ljcet3':3,'mbgfitn6':4,'uc64f2qs':5,'vtnag4op':6,'w3vcokrg':7,'uywljq4v':8,'ba56rk8v':9}
    reverse_dict = {v: k for k, v in class4_shixun_dict.items()}
    seq_id = [class4_shixun_dict[x] for x in seq]
    target_id = class4_shixun_dict[target]
    path_id = load(saved_4_path, int(user_id), seq_id, target_id)
    path = [reverse_dict[x] for x in path_id]
    return path
'''

@app.route('/path_rec', methods=['POST'])
def path_rec():
    result = {}
    user_id = request.form.get('user_id', type=str, default='')    
    class_id = request.form.get('class_id', type=str, default='')
    scene = request.form.get('scene', type=str, default='初始')    
    seq = request.form.get('seq', type=str, default='').split(',')  
    target_item_id = request.form.get('target', type=str, default='')    

    if user_id.strip() == '':
        result = {
            "status_code": str('False'),
            "error_msg": str('参数错误: 缺少user_id')
        }
        return json.dumps(result, ensure_ascii=False)  

    if class_id == '3':
        path = class3_path(user_id, scene, seq, target_item_id)
    elif class_id == '4':
        path = class4_path(user_id, scene, seq, target_item_id)

    result = {
        "status_code": str('True'),
        "user_id": str(user_id),
        "scene": str(scene),
        "recommend_count": str(len(path)),
        "recommend_results": path,
    }
    # return json.dumps(result, ensure_ascii=False)
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True, use_reloader=False)