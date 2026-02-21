import requests
import re
import os
import sys

FILE_ID = '17oxw9VD5g6Adulq2T1BD4RghktjIwsKi'
OUTPUT_FILE = 'potsdam_data.zip'

def download_from_gdrive(file_id, output_path):
    session = requests.Session()
    
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = session.get(url, stream=True)
    
    # Check for virus warning page and extract confirm token
    content_type = response.headers.get('Content-Type', '')
    
    if 'text/html' in content_type:
        # Need to get confirmation token
        html = response.text
        
        # Try multiple patterns to get the confirm token
        patterns = [
            r'confirm=([A-Za-z0-9_\-]+)',
            r'"confirm":"([^"]+)"',
            r'confirm=t&amp;',
        ]
        
        token = None
        for pattern in patterns:
            match = re.search(pattern, html)
            if match and len(match.groups()) > 0:
                token = match.group(1)
                break
        
        if token:
            print(f'Found confirm token: {token}')
            download_url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm={token}'
        else:
            # Use the direct confirm=t trick
            print('Using confirm=t method...')
            download_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    else:
        download_url = url
    
    print(f'Downloading from: {download_url}')
    response = session.get(download_url, stream=True)
    
    total = int(response.headers.get('Content-Length', 0))
    print(f'File size: {total / (1024**3):.2f} GB' if total > 0 else 'File size: unknown')
    
    downloaded = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    print(f'\rDownloaded: {downloaded/(1024**2):.1f} MB / {total/(1024**2):.1f} MB ({pct:.1f}%)', end='', flush=True)
    
    print(f'\nDownload complete! Saved to: {output_path}')
    print(f'File size on disk: {os.path.getsize(output_path) / (1024**3):.2f} GB')
    return output_path

download_from_gdrive(FILE_ID, OUTPUT_FILE)
