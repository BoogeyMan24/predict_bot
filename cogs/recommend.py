import pandas as pd
import random
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import discord
from discord.ext import commands
from fuzzywuzzy import process
import os
from openai import OpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPEN_AI_TOKEN')
OPEN_AI_ASSISTANTS_ID = os.getenv('OPEN_AI_ASSISTANT_ID')

client = OpenAI(
    api_key=OPEN_AI_API_KEY,
)


class RatingChoice(discord.ui.View):
    def __init__(self) -> None:
        super().__init__()
        self.value = None

    @discord.ui.button(label="1", style=discord.ButtonStyle.blurple)
    async def choose1(
            self, button: discord.ui.Button, interaction: discord.Interaction
    ) -> None:
        self.value = "1"
        self.stop()

    @discord.ui.button(label="2", style=discord.ButtonStyle.blurple)
    async def choose2(
            self, button: discord.ui.Button, interaction: discord.Interaction
    ) -> None:
        self.value = "2"
        self.stop()

    @discord.ui.button(label="3", style=discord.ButtonStyle.blurple)
    async def choose3(
            self, button: discord.ui.Button, interaction: discord.Interaction
    ) -> None:
        self.value = "3"
        self.stop()

    @discord.ui.button(label="4", style=discord.ButtonStyle.blurple)
    async def choose4(
            self, button: discord.ui.Button, interaction: discord.Interaction
    ) -> None:
        self.value = "4"
        self.stop()

    @discord.ui.button(label="5", style=discord.ButtonStyle.blurple)
    async def choose5(
            self, button: discord.ui.Button, interaction: discord.Interaction
    ) -> None:
        self.value = "5"
        self.stop()


# Main recommend COG class
class Recommend(commands.Cog, name="recommend"):
    def __init__(self, bot) -> None:
        self.bot = bot

        # Initialize data
        self.data_file = 'ml-100k/u.data'
        self.user_file = 'ml-100k/u.user'
        self.item_file = 'ml-100k/u.item'

        # Load data
        self.user_id_mapping, self.username_mapping = self.load_users()
        self.user_ratings = self.load_ratings()
        self.movie_titles = self.load_movie_titles()
        self.movie_mapping = self.load_movie_mapping()
        self.user_info = self.load_user_info()
        self.movie_info = self.load_movie_info()
        self.train_set, self.test_set, self.data = self.load_data()

        # Initialize and train model
        self.algo = SVD()
        self.retrain_model()

    # Init fucntion to load all ratings for each user
    def load_ratings(self):
        if not os.path.exists(self.data_file):
            return {}, {}

        column_names = ['user_id', 'movie_id', 'rating', 'other']
        user_data = pd.read_csv(self.data_file, delimiter='\t', names=column_names)

        user_ratings = {}
        for user_id, movie_id, rating, other in user_data.values:
            if user_ratings.get(user_id, "null") != "null":
                current = user_ratings[user_id]
                current.append({'movie_id': movie_id, 'rating': rating})
                user_ratings[user_id] = current

            else:
                user_ratings[user_id] = []
                user_ratings[user_id].append({'movie_id': movie_id, 'rating': rating})

        return user_ratings

    # Init function to load all registered users with the bot
    def load_users(self):
        if not os.path.exists(self.user_file):
            return {}, {}

        column_names = ['user_id', 'age', 'gender', 'occupation', 'discord_username', 'discord_user_id']
        user_data = pd.read_csv(self.user_file, delimiter='|', names=column_names)
        id_mapping = {str(row['discord_user_id']): str(row['user_id']) for _, row in user_data.iterrows() if
                      pd.notna(row['discord_user_id'])}
        name_mapping = {row['discord_username']: str(row['user_id']) for _, row in user_data.iterrows() if
                        pd.notna(row['discord_username'])}
        return id_mapping, name_mapping

    # Loads all movie titles from the data file
    def load_movie_titles(self):
        item_data = pd.read_csv(self.item_file, delimiter='|', encoding='ISO-8859-1',
                                usecols=[0, 1], names=['movie_id', 'title'])
        return dict(zip(item_data['title'], item_data['movie_id']))

    def load_movie_mapping(self):
        item_data = pd.read_csv(self.item_file, delimiter='|', encoding='ISO-8859-1',
                                usecols=[0, 1], names=['movie_id', 'title'])
        return dict(zip(item_data['movie_id'], item_data['title']))

    def load_user_info(self):
        item_data = pd.read_csv(self.user_file, delimiter='|', encoding='ISO-8859-1',
                                usecols=[0, 1, 2, 3], names=['user_id', 'age', 'gender', 'occupation'])
        return item_data

    def load_movie_info(self):
        item_data = pd.read_csv(self.item_file, delimiter='|', encoding='ISO-8859-1',
                                usecols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                         23],
                                names=[
                                    'movie_id',
                                    'title',
                                    'release',
                                    'url',
                                    'unknown',
                                    'action',
                                    'adventure',
                                    'animation',
                                    'childrens',
                                    'comedy',
                                    'crime',
                                    'documentary',
                                    'drama',
                                    'fantasy',
                                    'film-noir',
                                    'horror',
                                    'musical',
                                    'mystery',
                                    'romance',
                                    'sci-fi',
                                    'thriller',
                                    'war',
                                    'western',
                                ])
        return item_data

    # Loads all the data from the dataset
    def load_data(self):
        reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
        data = Dataset.load_from_file(self.data_file, reader=reader)
        train_set, test_set = train_test_split(data, test_size=0.25)
        return train_set, test_set, data

    # Used to retrain the model when new data is added by users.
    def retrain_model(self):
        self.algo.fit(self.train_set)

    # Function to add a user to the data
    async def add_user(self, discord_user):
        discord_user_id = str(discord_user.id)
        discord_username = discord_user.name

        if discord_username in self.username_mapping:
            return False, f"Discord username '{discord_username}' is already registered with ID {self.username_mapping[discord_username]}."

        new_user_id = max([int(uid) for uid in self.user_id_mapping.values() if uid.isdigit()], default=0) + 1
        self.user_id_mapping[discord_user_id] = str(new_user_id)
        self.username_mapping[discord_username] = str(new_user_id)

        # Format the new user data to match the existing structure of the u.user file
        new_user_data = f"\n{new_user_id}|M|other|00000|{discord_username}|{discord_user_id}\n"

        with open(self.user_file, 'a') as f:
            f.write(new_user_data)

        return True, f"Discord username '{discord_username}' added with ID {new_user_id}."

    async def get_movie_info(self, movie_title: str):
        movie_info_raw = None
        if len(self.movie_info.loc[self.movie_info["title"] == str(movie_title)]) == 0:
            return None
        else:
            movie_info_raw = self.movie_info.loc[self.movie_info["title"] == str(movie_title)].values[0]

        movie_info = {
            'movie_id': movie_info_raw[0],
            'title': movie_info_raw[1],
            'release': movie_info_raw[2],
            'url': movie_info_raw[3],
            'unknown': movie_info_raw[4],
            'action': movie_info_raw[5],
            'adventure': movie_info_raw[6],
            'animation': movie_info_raw[7],
            'childrens': movie_info_raw[8],
            'comedy': movie_info_raw[9],
            'crime': movie_info_raw[10],
            'documentary': movie_info_raw[11],
            'drama': movie_info_raw[12],
            'fantasy': movie_info_raw[13],
            'film-noir': movie_info_raw[14],
            'horror': movie_info_raw[15],
            'musical': movie_info_raw[16],
            'mystery': movie_info_raw[17],
            'romance': movie_info_raw[18],
            'sci-fi': movie_info_raw[19],
            'thriller': movie_info_raw[20],
            'war': movie_info_raw[21],
            'western': movie_info_raw[22],
        }

        return movie_info

    # Function to add a rating to a given movie name, requires a user to already be added to the database
    async def add_rating(self, discord_user, partial_movie_title: str, rating: float):
        discord_username = discord_user.name

        if discord_username not in self.username_mapping:
            return False, "Discord user not found. Please register first."

        user_id = self.username_mapping[discord_username]

        # Find the closest match for the movie title
        closest_match = process.extractOne(partial_movie_title, self.movie_titles.keys(), score_cutoff=70)
        if not closest_match:
            return False, "No close match found for the movie title. Please try again."

        movie_title, movie_id = closest_match[0], self.movie_titles[closest_match[0]]

        with open(self.data_file, 'a') as f:
            f.write(f"{user_id}\t{movie_id}\t{rating}\t0\n")

        # Update the data and retrain the model
        self.data = self.load_data()
        self.retrain_model()

        return True, f"Rating added for Discord user '{discord_username}' on movie '{movie_title}'."

    # Executes the discord command to add a user
    @commands.hybrid_command(
        name="add_user",
        description="Register the Discord user in the recommendation system.",
    )
    async def add_user_command(self, ctx: commands.Context):
        success, message = await self.add_user(ctx.author)
        await ctx.send(message)

    # Executes the discord command to add a rating, requires user to already be added
    @commands.hybrid_command(
        name="add_rating",
        description="Add a movie rating for a Discord user.",
    )
    async def add_rating_command(self, ctx: commands.Context, movie_title: str, rating: float):
        success, message = await self.add_rating(ctx.author, movie_title, rating)
        await ctx.send(message)

    @commands.hybrid_command(
        name="movieinfo",
        description="Get information about a movie to make better predictions."
    )
    async def movieinfo(self, ctx: commands.Context, *, movie_title: str):
        movie_info = await self.get_movie_info(movie_title)

        if movie_info is None:
            await ctx.send("Failed to find that movie")

        genres = []
        for key, value in movie_info.items():
            if value == 1 and key != "movie_id":
                genres.append(key)

        movie_embed = discord.Embed(
            title="Info for " + movie_info['title'],
            description="This movie was released on " + movie_info['release'] + "\nGenres: " + ", ".join(genres),
            color=0xBEBEFE
        )

        message = await ctx.send(embed=movie_embed)

    @commands.hybrid_command(name="predict", description="Try to predict a rating for a user closer than the ML.")
    async def predict(self, context: commands.Context):
        rating_buttons = RatingChoice()

        user_id, movie_id, rating = random.choice(self.test_set)

        embed = discord.Embed(
            title="User's past ratings",
            description="User " + str(user_id),
            color=0xBEBEFE
        )

        gender = "M"
        if self.user_info.loc[self.user_info["user_id"] == int(user_id)].values[0][2] == "M":
            gender = "Male"
        elif self.user_info.loc[self.user_info["user_id"] == int(user_id)].values[0][2] == "F":
            gender = "Female"
        embed.add_field(name="Age", value=self.user_info.loc[self.user_info["user_id"] == int(user_id)].values[0][1])
        embed.add_field(name="Gender", value=gender)
        embed.add_field(name="Occupation",
                        value=self.user_info.loc[self.user_info["user_id"] == int(user_id)].values[0][3])

        index = 0
        for ratingIn in self.user_ratings[int(user_id)]:
            if index == 21:
                break
            if ratingIn["movie_id"] != movie_id:
                embed.add_field(name=self.movie_mapping[int(ratingIn["movie_id"])], value=ratingIn["rating"])
            index += 1

        embed.set_footer(text="Don't know the movie? Get a quick description using !!movieinfo [moviename]")
        messageOld = await context.send(embed=embed)

        movie_info = await self.get_movie_info(
            self.movie_info.loc[self.movie_info["movie_id"] == int(movie_id)].values[0][1])

        genres = []
        for key, value in movie_info.items():
            if value == 1 and key != "movie_id":
                genres.append(key)

        rating_embed = discord.Embed(
            title="How will the user rate " + self.movie_mapping[int(movie_id)] + "?",
            description="This movie was released " + movie_info['release'] + "\nGenres: " + ", ".join(genres),
            color=0xBEBEFE
        )

        message = await context.send(embed=rating_embed, view=rating_buttons)

        await rating_buttons.wait()

        prediction = self.algo.predict(str(user_id), str(movie_id))

        won = abs(int(rating_buttons.value) - int(rating)) < abs(prediction.est - int(rating))

        if won:
            new_embed = discord.Embed(
                title="You were closer!",
                description="The correct value is " + str(rating) +
                            "\nYou chose " + rating_buttons.value +
                            "\n And the ML chose " + str(prediction.est) +
                            "\n You were " + str(abs(int(rating_buttons.value) - rating)) + " far from correct",
                color=0x00FF00
            )

            await message.edit(embed=new_embed, view=None, content=None)
        else:
            new_embed = discord.Embed(
                title="You lost!",
                description="The correct value is " + str(rating) +
                            "\nYou chose " + rating_buttons.value +
                            "\n And the ML chose " + str(prediction.est) +
                            "\n You were " + str(abs(int(rating_buttons.value) - rating)) + " far from correct",
                color=0xFF0000
            )

            await message.edit(embed=new_embed, view=None, content=None)

    # Executes the discord command to provide a recommendation to the user based on a movie they appear to be asking about, requires user to already be registered via the add_user command
    @commands.hybrid_command(
        name="recommend",
        description="Get recommendations based on username and partial movie name.",
    )
    async def recommend(self, ctx: commands.Context, *, partial_movie_name: str):
        discord_username = ctx.author.name

        if discord_username not in self.username_mapping:
            await ctx.send("Discord user not found. Please register first.")
            return

        user_id = self.username_mapping[discord_username]

        client = OpenAI(
            api_key=OPEN_AI_API_KEY,
        )

        # Create a thread with the initial user message
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": partial_movie_name
                }
            ]
        )

        # Start a run with the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=OPEN_AI_ASSISTANTS_ID,
        )

        # Wait for the assistant's response
        assistant_response = await self.wait_for_response(thread.id)

        if not assistant_response:
            await ctx.send("No response from the assistant. Please try again later.")
            return

        print("OpenAI's response from the bot: ", assistant_response[0].text.value)

        # Process the assistant's response
        closest_match = process.extractOne(assistant_response[0].text.value, self.movie_titles.keys(), score_cutoff=80)
        if not closest_match:
            embed = discord.Embed(
                title="No close match found for the movie name. Please try again.",
                color=0xE02B2B,
            )
            await ctx.send(embed=embed)
            return

        movie_name, movie_id = closest_match[0], self.movie_titles[closest_match[0]]
        prediction = self.algo.predict(str(user_id), str(movie_id))

        embed = discord.Embed(
            title=f"Closest match: '{movie_name}'",
            description=f"Prediction for User '{discord_username}' on Movie '{movie_name}':\nRating Prediction: {prediction.est}",
            color=0x57F287,
        )

        await ctx.send(embed=embed)

    async def wait_for_response(self, thread_id):
        """Wait for the assistant's response in the given thread."""
        for _ in range(30):  # Wait up to 30 seconds for a response
            await asyncio.sleep(1)  # Correctly await the sleep
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            if len(messages.data) > 1:  # Assuming the first message is the user's and the second is the assistant's
                return messages.data[0].content  # Return the assistants content


async def setup(bot) -> None:
    await bot.add_cog(Recommend(bot))
