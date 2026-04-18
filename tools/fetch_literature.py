import requests
import json
import os

def fetch_openalex_papers(query, max_results=5):
    print(f"Fetching OpenAlex papers for: {query}")
    # OpenAlex API documentation: https://docs.openalex.org/
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results,
        "sort": "cited_by_count:desc"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        for work in data.get('results', []):
            title = work.get('title', 'No Title')
            doi = work.get('doi', 'No DOI')
            publication_year = work.get('publication_year', '')
            authors = [author.get('author', {}).get('display_name', '') for author in work.get('authorships', [])]
            cited_by = work.get('cited_by_count', 0)
            
            results.append({
                'title': title,
                'authors': authors,
                'year': publication_year,
                'doi': doi,
                'citations': cited_by
            })
        return results
    else:
        print(f"Error fetching OpenAlex: {response.status_code}")
        return []

def main():
    queries = [
        "Stochastic Model Predictive Control",
        "Covariance-driven risk autonomous navigation",
        "Adaptive Hybrid LQR MPC"
    ]
    
    all_results = {}
    for q in queries:
        all_results[q] = fetch_openalex_papers(q)
        
    output_file = "docs/Research/fetched_literature.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Saved results to {output_file}")
    
if __name__ == '__main__':
    main()
