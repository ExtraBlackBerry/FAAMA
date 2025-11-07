import requests, os, json
import pandas as pd
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
                'created': pd.to_datetime(post['data']['created_utc'], unit='s'),
                'title': post['data']['title'],
                'text': post['data']['selftext'],
                'author': post['data']['author'],
                'upvotes': post['data']['ups'],
                'downvotes': post['data']['downs'],
                'upvote_ratio': post['data']['upvote_ratio'],
                'total_awards': post['data']['total_awards_received'],
                'score': post['data']['score'],
                'num_comments': post['data']['num_comments'],
                'type': post['data']['post_hint'] if 'post_hint' in post['data'] else 'text',
                # Could maybe add some like sentiment analysis wit subjectivity/polarity, idk might be slow or maybe do somewhere else
            }
            posts.append(post_data)
        return posts
    
    def get_post_comments(self, subreddit: str, post_id: str, limit: int = 10):
        response = requests.get(f'https://oauth.reddit.com/r/{subreddit}/comments/{post_id}',
                                headers=self.headers,
                                params={'limit': limit})
        return response.json()
    
    def extract_comment_data(self, comments_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        comments = []
        def extract_from_listing(listing_data):
            for child in listing_data.get('children', []):
                if child['kind'] == 't1':  # It's a comment
                    comment_data = child['data']
                    comments.append({
                        'id': comment_data.get('id'),
                        'created': pd.to_datetime(comment_data['created_utc'], unit='s'),
                        'body': comment_data.get('body'),
                        'author': comment_data.get('author'),
                        'score': comment_data.get('score'),
                        'ups': comment_data.get('ups'),
                        'downs': comment_data.get('downs'),
                        'total_awards': comment_data.get('total_awards_received'),
                        'parent_id': comment_data.get('parent_id'),
                        'depth': comment_data.get('depth'),
                    })
                    
                    # Recursively process replies if they exist
                    replies = comment_data.get('replies')
                    if replies and isinstance(replies, dict):
                        extract_from_listing(replies['data'])
        
        # Skip first element (post data), process second element (comments)
        if len(comments_json) > 1:
            extract_from_listing(comments_json[1]['data'])
        
        return comments
    
    def create_text_post(self, subreddit: str, title: str, text: str) -> Dict[str, Any]:
        post_data = {
            'title': title,
            'sr': subreddit,
            'kind': 'self',
            'text': text,
        }
        response = requests.post('https://oauth.reddit.com/api/submit',
                                 headers=self.headers,
                                 data=post_data)
        return response.json()
    
    def create_comment(self, parent_id: str, text: str):
        """Create a comment on a post or another comment."""
        data = {
            'thing_id': parent_id,
            'text': text
        }
        response = requests.post('https://oauth.reddit.com/api/comment',
                                headers=self.headers,
                                data=data)
        return response.json()
    
    def vote_post(self, post_id: str, direction: str) -> bool:
        """Vote on a post. Direction can be 'upvote', 'downvote', or 'unvote'."""
        dir_map = {'upvote': 1, 'downvote': -1, 'unvote': 0}
        if direction not in dir_map:
            raise ValueError("Direction must be one of 'upvote', 'downvote', or 'unvote'")
        data = {
            'id': post_id,
            'dir': dir_map[direction]
        }
        response = requests.post('https://oauth.reddit.com/api/vote',
                                 headers=self.headers,
                                 data=data)
        return response.status_code == 200  # Return True if vote was successful, maybe do this for other create methods too?
    
    def delete_post_or_comment(self, id: str) -> bool:
        """Delete a post or comment by its ID."""
        data = {
            'id': id
        }
        response = requests.post('https://oauth.reddit.com/api/del',
                                 headers=self.headers,
                                 data=data)
        return response.status_code == 200  # Return True if deletion was successful

# TESTING
if __name__ == "__main__":
    reddit = RedditInterface()          
    posts = reddit.get_subreddit_posts('NewZealand', limit=20, sort='hot')
    # print(json.dumps(posts, indent=4))
    extracted_posts = reddit.extract_post_data(posts)
    df = pd.DataFrame(extracted_posts)
    print(df.head(20))
    
    # Get comments for the first post
    first_post_id = extracted_posts[0]['id']
    comments_json = reddit.get_post_comments('NewZealand', first_post_id, limit=10)
    # print(json.dumps(comments_json, indent=4))
    extracted_comments = reddit.extract_comment_data(comments_json)
    comments_df = pd.DataFrame(extracted_comments)
    print(comments_df.head(10))
    