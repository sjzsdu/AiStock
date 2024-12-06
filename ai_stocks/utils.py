import hashlib

def generate_short_md5(input_string, len = 8):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()[:len]