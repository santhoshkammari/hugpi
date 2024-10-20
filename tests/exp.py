import time

from src.hugpi import HUGPIClient
from src.hugpi.features import HugpiInternetExplorer

if __name__ == '__main__':
    start_time = time.time()
    client = HUGPIClient(api_key="backupsanthosh1@gmail.com_SK99@pass")
    analyzer = HugpiInternetExplorer(client=client)
    query = "when is kohli born?"
    for x in analyzer.answer_query(query,stream=True):
        print(x.content[0]['text'],end="",flush=True)
    end_time = time.time()
    print(f"Summarization time: {end_time - start_time:.2f} seconds")