import requests, os, json
import pandas as pd
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()
class RedditInterface:
    def __init__(self):
        # Get access token for sending requests
        self.auth = requests.auth.HTTPBasicAuth(os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_SECRET_KEY")) # type: ignore
        self.login_data = {
            'grant_type': 'password',
            'username': os.getenv("REDDIT_USERNAME"),
            'password': os.getenv("REDDIT_PASSWORD"),
            'scope': 'identity read submit vote edit history'
        }
        self.headers = {'User-Agent': 'FaamaAPI/0.0.1'}
        self.token_response = requests.post('https://www.reddit.com/api/v1/access_token',
                                 auth=self.auth, data=self.login_data, headers=self.headers)
        self.token = self.token_response.json()['access_token']
        self.headers = {**self.headers, **{'Authorization': f'bearer {self.token}'}} # Update headers with token
        
    def get_subreddit_posts(self, subreddit: str, post_limit: int = 0, sort : str = 'hot'):
        """Get posts from a subreddit. Sort can be 'hot', 'new', 'top', or 'rising'."""
        if sort not in ['hot', 'new', 'top', 'rising']:
            raise ValueError("Sort must be one of 'hot', 'new', 'top', or 'rising'")
        response = requests.get(f'https://oauth.reddit.com/r/{subreddit}/{sort}',
                                headers=self.headers,
                                params={'limit': post_limit if post_limit > 0 else 100}) # Max limit is 100
        return response.json() # Bunch of stuff in here
    
    def get_post_comments(self, subreddit: str, post_id: str, comment_limit: int = 0):
        """Get comments for a specific post by its ID."""
        response = requests.get(f'https://oauth.reddit.com/r/{subreddit}/comments/{post_id}',
                                headers=self.headers,
                                params={'limit': comment_limit if comment_limit > 0 else 100})
        return response.json()
    
    def get_my_posts(self, limit: int = 25, sort: str = 'new') -> Dict[str, Any]:
        """Get posts made by bot. Can be sorted, best to use 'new' for most recent."""
        if sort not in ['hot', 'new', 'top', 'controversial']:
            raise ValueError("Sort must be one of 'hot', 'new', 'top', or 'controversial'")
        
        username = requests.get('https://oauth.reddit.com/api/v1/me', headers=self.headers).json()['name']
        response = requests.get(f'https://oauth.reddit.com/user/{username}/submitted',
                                headers=self.headers,
                                params={'limit': limit, 'sort': sort})
        return response.json()
    
    def get_my_comments(self, limit: int = 25, sort: str = 'new') -> Dict[str, Any]:
        """Get comments made by bot. Can be sorted, best to use 'new' for most recent."""
        if sort not in ['hot', 'new', 'top', 'controversial']:
            raise ValueError("Sort must be one of 'hot', 'new', 'top', or 'controversial'")
        
        username = requests.get('https://oauth.reddit.com/api/v1/me', headers=self.headers).json()['name']
        response = requests.get(f'https://oauth.reddit.com/user/{username}/comments',
                                headers=self.headers,
                                params={'limit': limit, 'sort': sort})
        return response.json()
    
    def extract_post_data(self, post_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant data for posts from the JSON returned by get_subreddit_posts."""
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
    
    def format_post(self, post_json: Dict[str, Any]) -> str:
        """Pass in the JSON of a single post to get a formatted string, good for displaying."""
        title = post_json['data']['children'][0]['data']['title']
        body = post_json['data']['children'][0]['data']['selftext']
        post_type = post_json['data']['children'][0]['data'].get('post_hint', 'text')
        author = post_json['data']['children'][0]['data']['author']
        created_utc = pd.to_datetime(post_json['data']['children'][0]['data']['created_utc'], unit='s')
        subreddit = post_json['data']['children'][0]['data']['subreddit']
        score = post_json['data']['children'][0]['data']['score']
        comments_count = post_json['data']['children'][0]['data']['num_comments']

        formatted_post = f"Title: {title}\nType: {post_type}\nBody: {body if body else 'No body text'}\nAuthor: {author}\nCreated: {created_utc}\nSubreddit: {subreddit}\nScore: {score}\nComments: {comments_count}"
        return formatted_post
    
    def extract_comment_data(self, comments_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relevant data for comments from the JSON returned by get_post_comments."""
        comments = []
        def extract_comment_or_reply(data):
            """Recursively extract comments/replies from a some json post or comment data."""
            for child in data.get('children', []):
                if child['kind'] == 't1':  # It's a comment
                    comment_data = child['data']
                    # Extracting this info for each comment/reply
                    comments.append({
                        'parent_id': comment_data.get('link_id'),
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
                    # Keep getting nested replies until there are no more
                    replies = comment_data.get('replies')
                    if replies and isinstance(replies, dict):
                        extract_comment_or_reply(replies['data'])
        if len(comments_json) > 1:
            extract_comment_or_reply(comments_json[1]['data']) # Comments are in the second element
        return comments
    
    def extract_my_comment_data(self, comments_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comments made by the bot from json returned by get_my_comments.
        This is different from extract_comment_data since self comment structure is different."""
        comments = []
        for comment in comments_json['data']['children']:
            comment_data = comment['data']
            comments.append({
                'parent_id': comment_data.get('link_id'),
                'id': comment_data['id'],
                'created': pd.to_datetime(comment_data['created_utc'], unit='s'),
                'body': comment_data['body'],
                'subreddit': comment_data['subreddit'],
                'link_title': comment_data.get('link_title'),
                'score': comment_data['score'],
                'ups': comment_data['ups'],
                'downs': comment_data['downs'],
                'parent_id': comment_data['parent_id'],
                'link_id': comment_data['link_id'],
            })
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
        """Create a comment on a post or another reply to another comment."""
        data = {
            'thing_id': parent_id,
            'text': text
        }
        response = requests.post('https://oauth.reddit.com/api/comment',
                                headers=self.headers,
                                data=data)
        return response.json()
    
    def vote_post(self, post_id: str, direction: str) -> bool: #TODO: Change to thing_id to allow voting on comments too
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
    # Setup
    retrieving = True
    posting = False
    deleting = True
    reddit = RedditInterface()
    
    # ========================================
    # RETRIEVING POSTS
    # ========================================
    if retrieving:
        print("=" * 50)
        print("RETRIEVING POSTS")
        print("=" * 50)
        
        posts = reddit.get_subreddit_posts('AskReddit', post_limit=20, sort='hot')
        extracted_posts = reddit.extract_post_data(posts)
        df = pd.DataFrame(extracted_posts)
        print(df.head(20))
        
        # ========================================
        # RETRIEVING COMMENTS
        # ========================================
        print("\n" + "=" * 50)
        print("RETRIEVING COMMENTS FOR FIRST POST")
        print("=" * 50)
        
        first_post_id = extracted_posts[0]['id']
        comments_json = reddit.get_post_comments('AskReddit', first_post_id, comment_limit=10)
        extracted_comments = reddit.extract_comment_data(comments_json)
        comments_df = pd.DataFrame(extracted_comments)
        print(comments_df.head(10))
        
        # ========================================
        # FORMATTING POST
        # ========================================
        print("\n" + "=" * 50)
        print("FORMATTED POST")
        print("=" * 50)
        
        formatted_post = reddit.format_post(posts)
        print(formatted_post)
    
    # ========================================
    # CREATING POST, COMMENT, AND VOTING
    # ========================================
    if posting:
        print("\n" + "=" * 50)
        print("CREATING POST")
        print("=" * 50)
        
        # Create a text post
        title = "Test Post from API"
        content = "This is a test post created via the Reddit API."
        post = reddit.create_text_post('test', title, content)
        
        if post.get('success'):
            # Extract post URL from the jQuery response
            post_url = None
            for item in post.get('jquery', []):
                # Look for redirect in the jQuery array, it contains the post URL
                if len(item) >= 4 and item[2] == "attr" and item[3] == "redirect":
                    idx = post['jquery'].index(item)
                    if idx + 1 < len(post['jquery']):
                        next_item = post['jquery'][idx + 1]
                        if next_item[2] == "call" and len(next_item[3]) > 0:
                            post_url = next_item[3][0]
                            break
            
            if post_url:
                # Extract post ID from URL
                post_id = post_url.split('/comments/')[1].split('/')[0]
                print(f"Post created successfully!")
                print(f"Post ID: {post_id}")
                print(f"Post URL: {post_url}")
                
                # ========================================
                # CREATING COMMENT
                # ========================================
                print("\n" + "=" * 50)
                print("CREATING COMMENT")
                print("=" * 50)
                
                comment = reddit.create_comment(f't3_{post_id}', "This is a test comment.")
                print("Comment created:", json.dumps(comment, indent=4))
                
                # ========================================
                # VOTING ON POST
                # ========================================
                print("\n" + "=" * 50)
                print("VOTING ON POST")
                print("=" * 50)
                
                vote_response = reddit.vote_post(f't3_{post_id}', 'upvote')
                print(f"Vote successful: {vote_response}")
            else:
                print("ERROR: Could not extract post URL from response")
        else:
            print("ERROR: Post creation failed")
            print(json.dumps(post, indent=4))
            
    # ========================================
    # DELETING POST OR COMMENT
    # ========================================
    if deleting:
        # =======================================
        # GET MY POSTS TO DELETE
        # =======================================
        print("\n" + "=" * 50)
        print("GETTING MY POSTS TO DELETE")
        print("=" * 50)

        my_posts = reddit.get_my_posts()
        my_posts_extracted = reddit.extract_post_data(my_posts)
        my_posts_df = pd.DataFrame(my_posts_extracted)
        print(my_posts_df.head(5))
        
        # =======================================
        # GET MY COMMENTS TO DELETE
        # =======================================
        print("\n" + "=" * 50)
        print("GETTING MY COMMENTS TO DELETE")
        print("=" * 50)
        my_comments = reddit.get_my_comments()
        my_comments_extracted = reddit.extract_my_comment_data(my_comments)
        print(pd.DataFrame(my_comments_extracted).head(5))

        # =======================================
        # GET SINGLE POST FROM A SUBREDDIT
        # =======================================

        posts = reddit.get_subreddit_posts('Pics', post_limit=5, sort='hot')
        print(json.dumps(posts['data']['children'][0], indent=4))
        
        # =======================================
        # TODO:
        # DELETE MOST RECENT POST
        # =======================================
        
        # ======================================
        # TODO:
        # DELETE MOST RECENT COMMENT
        # ======================================