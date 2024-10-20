import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from pyopengenai.researcher_ai import RealTimeGoogleSearchProvider
from ..model.api import HUGPIClient
from ..model.api.types._message import Message
from ..model.api.types._model_types import MODELS_TYPE
import logging
logger = logging.getLogger(__name__)


class HugpiInternetExplorer:
    def __init__(self, model_name: MODELS_TYPE = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                 api_key: str=None,
                 client = None,
                 debug = False):
        ## check for apikey or client
        if api_key is None and client is None:
            raise ValueError("Either API key or HuggingFace client should be provided.")
        logger.setLevel(level='DEBUG' if debug else 'INFO')
        self.search = RealTimeGoogleSearchProvider()
        self.client = HUGPIClient(model_name, api_key=api_key) if client is None else client
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.debug = debug

    @staticmethod
    def _fetch_url_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=5)
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    @staticmethod
    def _extract_content(html_content: str) -> Dict[str, List[str]]:
        soup = BeautifulSoup(html_content, 'html.parser')
        code_blocks = [code.get_text() for code in soup.find_all(['pre', 'code'])]
        for script in soup(["script", "style"]):
            script.decompose()
        text_content = [line.strip() for line in soup.get_text().split('\n') if line.strip()]
        return {"code": code_blocks, "text": text_content}

    @staticmethod
    def _segment_content(content: Dict[str, List[str]], max_length: int = 300) -> List[Tuple[str, str]]:
        segments = [('code', code) for code in content['code']]
        current_segment = ""
        for line in content['text']:
            if len(current_segment) + len(line) > max_length:
                if current_segment:
                    segments.append(('text', current_segment))
                current_segment = line
            else:
                current_segment += " " + line if current_segment else line
        if current_segment:
            segments.append(('text', current_segment))
        return segments

    def _calculate_relevance_score(self, segment: Tuple[str, str], query: str, is_code_query: bool) -> float:
        segment_type, content = segment
        query_words = query.lower().split()
        keyword_score = sum(content.lower().count(word) for word in query_words)
        tfidf_matrix = self.vectorizer.transform([query, content])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        keyword_weight, cosine_weight = (0.6, 0.4) if is_code_query and segment_type == 'code' else (0.4, 0.6)
        combined_score = (keyword_score * keyword_weight) + (cosine_sim * cosine_weight)
        if is_code_query and segment_type == 'code':
            combined_score *= 1.5
        return combined_score

    def _process_url(self, url: str, query: str, is_code_query: bool) -> List[Tuple[str, str, float]]:
        html_content = self._fetch_url_content(url)
        extracted_content = self._extract_content(html_content)
        segments = self._segment_content(extracted_content)
        scored_segments = []
        for segment in segments:
            score = self._calculate_relevance_score(segment, query, is_code_query)
            if score > 0:
                scored_segments.append((url, segment[1], score))
        return scored_segments

    @staticmethod
    def _is_code_query(query: str) -> bool:
        code_keywords = ['code', 'function', 'script', 'program', 'implement', 'git', 'github']
        return any(keyword in query.lower() for keyword in code_keywords)

    def _get_google_urls(self,query,max_urls):
        start = time.time()
        original_urls = self.search.perform_search(query, max_urls=max_urls)
        end = time.time()
        logger.debug(f"Search time: {end - start:.2f} seconds")
        return original_urls

    def _get_url_extract_content(self,urls,chunk_size:int=1000):
        start = time.time()
        sample_content = " ".join(self._fetch_url_content(urls[0]).split()[:chunk_size])
        end = time.time()
        logger.debug(f"URL ExtractFetch time: {end - start:.2f} seconds")
        return sample_content



    def _get_results(self, query: str, top_n: int = 4,max_urls=4,
                     chunk_size:int=1000) -> List[Tuple[str, str, float]]:
        start_time = time.time()
        code_query = self._is_code_query(query)
        original_urls = self._get_google_urls(query, max_urls)
        urls = list(set(original_urls))
        logger.debug(f"URLS: {urls}")
        sample_content = self._get_url_extract_content(urls,chunk_size=chunk_size)
        self.vectorizer.fit([query, sample_content])
        all_scored_segments = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._process_url, url, query, code_query): url for url in urls}
            for future in as_completed(future_to_url):
                all_scored_segments.extend(future.result())
        sorted_segments = sorted(all_scored_segments, key=lambda x: x[2], reverse=True)
        diverse_results = []
        seen_urls = set()
        for segment in sorted_segments:
            if segment[0] not in seen_urls:
                diverse_results.append(segment)
                seen_urls.add(segment[0])
            elif len([r for r in diverse_results if r[0] == segment[0]]) < 2:
                diverse_results.append(segment)
            if len(diverse_results) >= top_n:
                break
        end_time = time.time()
        logger.debug(f"Processing time: {end_time - start_time:.2f} seconds")
        return diverse_results

    @sleep_and_retry
    @limits(calls=5, period=1)
    def _rate_limited_api_call(self, url: str, content: str, query: str) -> str:
        results = f"<realtime_data>URL:{url}\nContent: {content}</realtime_data><query>{query}</query>"
        response = self.client.messages.create(messages=[
            {
                "role": "system",
                "content": "You are a content relevant filter AI. Summarize only content relevant to the query. If you are 100% sure it's not relevant, remove that info. Be careful with Python codes. Include URLs if needed."
            },
            {
                "role": "user",
                "content": results
            }
        ])
        return response.content[0]['text']

    def _process_api_url(self, args: Tuple[str, str, float, str]) -> str:
        url, content, score, query = args
        try:
            return self._rate_limited_api_call(url, content, query)
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None

    def _summarize_results(self, query: str, results: List[Tuple[str, str, float]]) -> List[str]:
        summarized_data = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._process_api_url, (url, content, score, query)): url for url, content, score in results}
            for future in tqdm(as_completed(future_to_url), total=len(results), desc="Summarizing", unit="url"):
                url = future_to_url[future]
                try:
                    summary = future.result()
                    if summary:
                        summarized_data.append(summary)
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
        end_time = time.time()
        logger.debug(f"Summarization time: {end_time - start_time:.2f} seconds")
        return summarized_data

    def answer_query(self, query: str,max_urls:int=4,chunk_size:int=1000,**kwargs) -> Message:
        results = self._get_results(query,max_urls=max_urls,
                                    chunk_size=chunk_size)
        summarized_data = self._summarize_results(query, results)
        summarized_data = "\n".join(summarized_data)
        return self.client.messages.create(messages=[
            {
                "role": "system",
                "content": "Answer the following question based on the realtime data"
            },
            {
                "role": "user",
                "content": f"<real_time_data>{summarized_data}</real_time_data>\n<query>{query}</query>"
            }
        ], **kwargs)

# Example usage

