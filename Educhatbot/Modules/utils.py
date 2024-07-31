import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import config 
import random
import re


def upload2s3(filepath, filename, s3name=config.S3BUCKNAME, extension="wav"):
        file_name = "kb-storage/" + filename  
        if not extension  is None : file_name += f".{extension}"
        else: file_name += ""
        
        try:
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            with open(filepath, 'rb') as data:
                s3.upload_fileobj(data, s3name, file_name) #upload to s3.

            text = "https://{}.s3.amazonaws.com/{}".format(s3name, file_name)
            return text
        except Exception as e:
            print("S3 error", e)
            return f"Error to save file {file_name} to S3" 
        
def downloadfolder4s3(s3_path, save_root, bucket_name=config.S3BUCKNAME, replace_word=['kb-storage/']):
    try: 
        s3 = boto3.resource('s3', config=AWSConfig(signature_version=UNSIGNED))
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_path):
            tmppath = str(obj.key)
            for w in replace_word: tmppath = tmppath.replace(w, "")
            tmppath = "/".join([p for p in tmppath.split("/") if p.find(".") < 0])
            tmppath = os.path.join(save_root, tmppath).replace("\\", "/")
            os.makedirs(tmppath,  exist_ok=True)
            s3.Object(bucket_name, obj.key).download_file(os.path.join(tmppath, obj.key.split("/")[-1]).replace("\\", "/"))
        return True 
    except:  False 


   
def get_domain(url):
    try:
        pattern = r'(https*://[^/]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else: 
            print("cannot find domain url of  ", url)
            raise Exception("not find domain url")
    except:
        return ""


def return_mediaURL(filename,  extension="wav", url="/"):
        url = get_domain(url)
        file_name =  filename + f".{extension}"
        url += "/audio/" + file_name
        return url

def find_json_text(text):
     
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    extracted_text = text[start_index:end_index]
    return extracted_text