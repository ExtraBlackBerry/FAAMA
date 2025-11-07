import requests, os, json
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List
class RedditInterface:
    def __init__(self):
        # Get access token
        self.auth = requests.auth.HTTPBasicAuth(os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_SECRET_KEY"))
        self.login_data = {
            'grant_type': 'password',
            'username': os.getenv("REDDIT_USERNAME"),
            'password': os.getenv("REDDIT_PASSWORD")}
        self.headers = {'User-Agent': 'FaamaAPI/0.0.1'}
        self.token_response = requests.post('https://www.reddit.com/api/v1/access_token',
                                 auth=self.auth, data=self.login_data, headers=self.headers)
        self.token = self.token_response.json()['access_token']
        self.headers = {**self.headers, **{'Authorization': f'bearer {self.token}'}} # Update headers with token
        
    def get_subreddit_posts(self, subreddit: str, limit: int = 10, sort : str = 'hot'):
        if sort not in ['hot', 'new', 'top', 'rising']:
            raise ValueError("Sort must be one of 'hot', 'new', 'top', or 'rising'")
        response = requests.get(f'https://oauth.reddit.com/r/{subreddit}/{sort}',
                                headers=self.headers,
                                params={'limit': limit})
        return response.json() # Bunch of stuff in here
    
    def extract_post_data(self, post_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        posts = []
        for post in post_json['data']['children']:
            post_data = {
                'subreddit': post['data']['subreddit'],
                'id': post['data']['id'],
                'title': post['data']['title'],
                'text': post['data']['selftext'],
                'author': post['data']['author'],
                'score': post['data']['score'],
                'upvotes': post['data']['ups'],
                'downvotes': post['data']['downs'],
                'upvote_ratio': post['data']['upvote_ratio'],
                'total_awards': post['data']['total_awards_received'],
                'num_comments': post['data']['num_comments'],
                'type': post['data']['post_hint'] if 'post_hint' in post['data'] else 'text',
                # Could maybe add some like sentiment analysis wit subjectivity/polarity, idk might be slow or maybe do somewhere else
            }
            posts.append(post_data)
        return posts

# TESTING
if __name__ == "__main__":
    reddit = RedditInterface()          
    posts = reddit.get_subreddit_posts('NewZealand', limit=20, sort='hot')
    print(json.dumps(posts, indent=4))
    extracted_posts = reddit.extract_post_data(posts)
    for post in extracted_posts:
        print(json.dumps(post, indent=4))