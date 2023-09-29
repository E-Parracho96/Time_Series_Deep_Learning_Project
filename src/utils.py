import os
import sys
import pickle
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise  CustomException(e,sys)
    
# Possível alternativa ao save_object, dai ter criado a pasta Notebook para guardar os pickle
def save_object2(obj):
    with open(os.path.join('Results','name_'+'.pickle'), 'wb') as file_obj:
        pickle.dump(obj, file_obj)
    return

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

