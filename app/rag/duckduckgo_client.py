"""
DuckDuckGo Search Client.
Provides keyless, zero-cost fallback search using direct HTML scraping.
More robust than the library for low-volume fallback.
"""

from typing import List, Dict
import httpx
import re
import logging
from urllib.parse import unquote

logger = logging.getLogger(__name__)

class DuckDuckGoClient:
    def __init__(self):
        self.base_url = "https://lite.duckduckgo.com/lite/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://lite.duckduckgo.com/"
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform search via DuckDuckGo Lite endpoint.
        Provides zero-cost fallback search without API keys.
        """
        if not query:
            return []

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                data = {"q": query}
                resp = client.post(self.base_url, data=data, headers=self.headers)
                
                if resp.status_code != 200:
                    logger.warning(f"DDG Lite Search failed: Status {resp.status_code}")
                    return []
                
                if "CAPTCHA" in resp.text:
                    logger.error("DDG Lite Search blocked by CAPTCHA.")
                    return []
                
                return self._parse_lite_html(resp.text, max_results)
                
        except Exception as e:
            logger.warning(f"DDG connection error: {e}")
            return []

    def _parse_lite_html(self, html: str, limit: int) -> List[Dict]:
        """Parse Lite HTML results using regex."""
        results = []
        
        # Capture links: rel="nofollow" href="..." class='result-link'
        link_pattern = re.compile(
            r"<a[^>]*href=\"([^\"]+)\"[^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>", 
            re.IGNORECASE | re.DOTALL
        )
        # Capture snippets: td class='result-snippet'
        snippet_pattern = re.compile(
            r"<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>", 
            re.IGNORECASE | re.DOTALL
        )
        
        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)
        
        # Zip them together (they appear in pairs in the HTML)
        for (href, title), snippet_raw in zip(links, snippets):
            if len(results) >= limit:
                break
                
            try:
                # 1. URL Decoding
                url = unquote(href)
                
                # Filter out ads that redirect through DDG if they seem like garbage
                # though some ads are actually useful. We'll keep them if they look like real URLs.
                if "duckduckgo.com/y.js" in url:
                    # Try to extract the real URL if it exists in the query params
                    if "u3=" in url:
                        # Find the u3 param
                        match = re.search(r'u3=(.*?)&', url)
                        if match:
                            url = unquote(match.group(1))
                        else:
                            # Might be at the end
                            match = re.search(r'u3=(.*?)$', url)
                            if match:
                                url = unquote(match.group(1))

                # 2. Cleaning HTML tags
                def clean(t):
                    return re.sub(r'<[^>]+>', '', t).strip()
                
                final_title = clean(title)
                final_snippet = clean(snippet_raw)
                
                # 3. Final verification
                if not final_title or not final_snippet:
                    continue
                    
                results.append({
                    "title": final_title,
                    "url": url,
                    "content": final_snippet,
                    "source": "duckduckgo"
                })
                
            except Exception:
                continue
                
        return results

def get_ddg_client() -> DuckDuckGoClient:
    return DuckDuckGoClient()
