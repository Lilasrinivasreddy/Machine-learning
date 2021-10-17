# Movie recommendation System using colloborative filtering 
## Skills required/improved during project:
* `1.Python 3`     
* `2.Flask`

## Overview
* `Developed a user filtering based Movie Recommendation system using movielens dataset from from grouplens.org.`     
* `Deploy the model using Flask based web server`
* `Connected the flask based app to movie posters dataset (collected with help of url and imdb id dataset) `


<p align='center'>
	<img src="dataset/readme images/Webapp.png" width=800 >
</p>

## Approach & Methodology

<b>Collaborative filtering</b><br><br>

Collaborative filtering is used to tailor recommendations based on the behavior of persons with similar interests. Sometimes it can be based on an item bought by the user. Since this method does not require a person himself to always contribute to a data store, and voids can be filled by the actions of other persons/ actions by the same person on other items.

There are 4 approaches in collaborative fltering but i go for the following one: 
<b>Item — to — Item approach</b><br>
Instead of using ratings given by the users to calculate neighborhood, the ratings are used to find similarity between items(movies). The same pearson coefficient can be used here to calculate similarity.

<p align='center'>
	<img src="dataset/readme images/method.jpg" width=800 >
</p> 

## Using web app
First field require to enter the "Name of the movie" .<br>
Second field require to enter the rating of the moving.Then recommender populate the recommended movies result.<br>


