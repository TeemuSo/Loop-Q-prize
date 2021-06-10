import boto3
import botocore
from dotenv import dotenv_values, find_dotenv, load_dotenv

bucket = 'loopqprize'

try:
    s3 = boto3.resource('s3')

    try:
        s3.meta.client.head_bucket(Bucket=bucket)
        print("Bucket connection stable!\n")
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 403:
            print("Private Bucket. Forbidden Access!")
        elif error_code == 404:
            print("Bucket Does Not Exist! Contact the project owner!")

except KeyError as e:
    print(".env must be configured!")
    print(f"Missing key: {e.args}")
