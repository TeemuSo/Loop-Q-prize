import sys

import boto3
import botocore

REQUIRED_PYTHON = "python3"


def _test_aws_connection():
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


def main():
    _test_aws_connection()
    
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")




if __name__ == '__main__':
    main()
