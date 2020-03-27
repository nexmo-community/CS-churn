Nexmo’s Conversation API allows developers to build their own Contact Center solution. With voice, text and integrations to other solutions, Nexmo allows you to build a complex solution, without the development being complex.

When building your own contact center solution, you need it to be smart. The most powerful solutions use AI to route calls, translate text, recommend products and so on. What's great is that you don’t have to be a AI researcher with a PHD to integrate AI into your application. You also don’t have to rely on a 3rd party system. This seems impossible at first, but there has been lots of progress on many machine learning libraries so that you as a developer can build a machine learning system into your solution.

In this post, we’ll look into adding a way to predict the likelihood of a customer churning. Churn, by definition is the number of customers who stopped using your product, divided by the number of total customers. For example, a company with a 1% churn per month with 1000 customers, means that 10 out of 1000 customers stop using the company's service. This can be measured by month, quarter  year and so on. It depends as a company how this is tracked.

A contact center is one place where customers interact directly with the business, especially with customer service. And losing a customer with bad support could have a large impact on churn and therefore, your health as a company.

We’ve built a simple demo application that simulates a conversation between a customer and agent, using the Conversation API.

![](images/Churn Demo.mp4)

# Overview
In our demo, we have 2 user personas, a customer and an agent. For this example, we’ll assume the company is a TV service provider, and the customer has a question about their service. We also assume that this customer has been with the company for awhile, and we have data to support this.
In our example, we have some data about the user. This would be how long in months that the customer has used the service, what their billing issue, what services they have and so on.

For our demo, when the customer interacts with the agent, we show the likelihood of the user churning on the agent’s screen, as soon as they begin to interact. This could be helpful to the agent before starting the conversation, so that more attention could be given to the customer, depending on the likelihood of churn.

# Prerequisites
- Nexmo account
- Access to [Google Colab](https://colab.research.google.com)

We'll be using [Hui Jing Chen](https://www.nexmo.com/blog/author/huijing)'s [blog post](https://www.nexmo.com/blog/2019/10/18/how-to-build-an-on-page-live-chat-dr) as a starter. We will be adding our churn prediction functionality on top of this application.
To run our application locally, we'll also need to use [ngrok](https://ngrok.com/). If you are not familiar with Ngrok, please refer to our [Ngrok tutorial](https://www.nexmo.com/blog/2017/07/04/local-development-nexmo-ngrok-tunnel-dr/) before proceeding.

Before going over the application, we first need to build our model. And to build it, we'll be using [Jupyter Notebooks](https://jupyter.org) running on Google Colab. A Jupyter Notebook is an interactive way to run code, and is widely used in data science and machine learning. Google Colab is a free service that lets you run these notebooks in the cloud. The code to build the notebook is located on [here](Coversation_Service_Churn_Prediction.ipynb).To run this notebook, you can upload to Google Colab.

For this tutorial, we will assume you have a basic understanding of what machine learning is, but you won't need to fully understand everything in order to follow along.

In order to build a model, we first need data. And for this example, we’ll use [Telecom Churn Dataset from IBM](https://www.kaggle.com/zagarsuren/telecom-churn-dataset-ibm-watson-analytics). This dataset contains 7043 rows of a telcom’s user data. This data is only used for examples and does not reflect a real company's users. Let's have a look with the first 10 rows of the data using Pandas. [Pandas](https://pandas.pydata.org) is a python library that is used for processing and understanding data. For each user, we have 23 columns, also known as features. These include the customer gender(male, female), tenure(how long have they been a customer) and if they have different services, including phone, internet and TV.
In order to build our model, we first need to make sure there are no empty values in the dataset. If we don't check for this and try to build our model, we will have errors.

Let’s read in our dataset and remove any empty values.
```python
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.dropna(axis='columns', inplace=True)
```
Next, we can use `df.head()` to view the first 10 rows of the data. When we look at this data, we see that many of the rows contain strings. (`YES`, `NO`). We now have to convert these strings into numbers, because machine learning models only know how to deal with numbers.
For each row that contains a string, we need to see if the strings are unique. To view all the possible values in this column, pandas has a function, called `unique()` to do this for us.
```python
df.Partner.unique()
```
returns:
`array(['Yes', 'No'], dtype=object)`

This means that the row only contains the values `YES` and `NO`. For this column, we can convert those strings into booleans(1,0).
However, if we look at the other rows, say `PaymentMethod` There’s more values than `YES` or `NO`.

```python
df.PaymentMethod.unique()
```
returns
`array(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'], dtype=object)`

So for this column, we need to do a little more work. What we can do is that whenever a value is `YES` or `NO`, we can convert it to `1` or `0`, respectively. When it's any other string, let's set it to `-1`. Again, the machine learning model can only use numbers, so that’s why we set it to -1. It can be any number you like, but I think `-1` makes sense.
If we look at the other columns, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV` and `StreamingMovies`, they appear to be similar. So let's write a very simple function that goes through each column and converts our strings into ints.

```python
numeric_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
def to_numeric(s):
  if s == "Yes":
    return 1
  elif s == "No":
    return 0
  else: return -1

for feature in numeric_features:
  df[feature] = df[feature].apply(to_numeric)
```

`Numeric_features` is a list of all the columns that we need to update. `to_numeric` is a function that takes in the value from every row and converts the string to an int. Finally, we’ll loop through all the items in `to_numeric` and call the pandas function `apply` to call our function.
Let’s have a look at the first 10 rows to verify.

![DataFrame]("images/df_head.jpg")

It looks like those rows are now valid, but we have to deal with the other columns. Lets first inspect `Contact` and see what values are shown.
```python
df.Contract.unique()
```
which returns:
`array(['Month-to-month', 'One year', 'Two year'], dtype=object)`

These values are still strings, but it's not as easy as converting to `1`'s and `0`'s. There are other columns in this dataset that are similar, including `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling` and `PaymentMethod`.

Luckily, in pandas, we can convert these values using `get_dummies()`. This function, when applied to a column, will create a new column for every possible value. And every value in each of these new columns will be `1` or `0`. This is also known as [one-hot-encoding](https://www.ritchieng.com/machinelearning-one-hot-encoding/).
So for example, let’s take the `Contract` column, which contains the values of `Month-to-month`, `One year` and `Two year`. Using `get_dummies()`, we will create 3 new columns called `Contract_Month-to-month`, `Contract_One year` and `Contract_Two year`. And every value in these columns will be either `1` or `0`.

```python
categorical_features = [
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_features)
```
Here, we create a list of these features that we want to convert to categorical and call the `get_dummies` function using the dataFrame(`df`) and the list of columns(`categorical_features`).
Lets again view the first 10 rows to double check our work. Since the dataframe is now 41 features, we'll link to the cell [here](https://colab.research.google.com/drive/1e7uDkEkHyY-r6UKJh0ZvHBenhGOVjdCO#scrollTo=V3jFDzt6C9qa&line=1&uniqifier=1).

Looks like all the columns are numeric. Let's build our model.
To build our model, we'll use another package called [scikit-learn](https://scikit-learn.org/stable/).Scikit-Learn has many built it functions to process and train a model with our data.

First, we need 2 matrices, `X` and `y`. `X` is a matrix that includes all of our features except for the feature that we are using to make predictions(`Churn`). `Y` is just the value of `Churn`, which is a `1` or `0`.
```python
X = df.drop(labels='Churn',axis=1)
Y = df.Churn
print(X.shape, Y.shape)
```
This returns:
`((7043, 40), (7043,))`
Which means that there are 7043 rows and 40 columns in X. For Y, we have 7043 rows.
Next, we have to split our data into a train and test set. When training our model, we only use a portion of the dataset. The rest of the data is used for testing. This is to see how well our model has learned the data. A testing set contains about 70% of the data. Using scikit-learn, lets split our dataset using [train-test-split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)
```
Finally, we can build a model.
For this example, we will use a simple model. But in the real world, we need to test different models with our dataset to see what's best. This is where a lot of the work lies in machine learning. There are no hard or fast rules to selecting a model for your data. It always depends on the dataset.
For this example, we will use [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to train our model. It's a good model to start out with since it generally works well with making predictions for boolean values.
scikit-learn makes it very simple to implement this model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = LogisticRegression()
model.fit(X_train, y_train)
```
Once the model is fit, which means that the model was trained on the training data, we can now test it, using the predict function on the test data.
```python
model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))
````
using `metrics.accuracy_score` we can print out our accuracy, 0.8135352579271179 which is ~89%. This is means that when our model is given data to make a prediction, its result is correct about 89% percent of the time. Note, this accuracy is just one of many metrics when [evaluating a model](https://scikit-learn.org/stable/modules/model_evaluation.html).

Once we have our model, we'll need to save it in order to be used in our contact center app. To save we use the [`joblib` function dump()](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html) to save our model.
We will also need to save the names of the columns that we used for training. This will be used when we build our server to make predictions.

```Python
model_columns = list(df.columns)
model_columns.remove('Churn')
joblib.dump(model_columns, 'model_columns.pkl')
```
Next, we'll build a simple Python server to serve this model.

# Model Serving using Flask
Next, we will use [Flask](https://palletsprojects.com/p/flask/) to create a basic server application to host our model. We need a way to host and model and make predictions. If this was a production application, we would need to save user information, which would contain the same information that we used for training in our dataset. These would include how long the user has been with the company(tenure), whether they have `InternetService`, `PhoneService`, `OnlineBackup` etc...
This way, when we make our prediction, we would make a query to our database to return the users info, make a prediction from our model, and return the likelihood of churning. However, for this example, we will make up information for a user and send that prediction back to the application.

In `server.py`, we will make a simple Flask app, which only has one endpoint called `/predict`. This endpoint will simulate user's data, invoke our model and return the prediction as a JSON response.
Before making the prediction, we first need to load the saved model when the server starts.

```Python
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
  #will return prediction

if __name__ == '__main__':
     model = joblib.load('model/model.pkl')
     model_columns = joblib.load('model/model_columns.pkl')
     app.run()
```

Here we use `joblib` to load our model and the columns that we used for training. We need to make sure we have copied `model.pkl` and `model_columns.pkl` into our server application. In the `predict()` function, we'll generate a random user's data, create a new DataFrame from the returned data, using the saved columns names.
```Python

def predict():
    random_user_data = generate_data()
    #https://towardsdatascience.com/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa
    query = pd.get_dummies(pd.DataFrame(random_user_data, index=[0]))
    query = query.reindex(columns=model_columns, fill_value=0)

    #return prediction as probability in percent
    prediction = round(model.predict_proba(query)[:,1][0], 2)* 100
    return jsonify({'churn': prediction})
```

The `generate_data()` function creates a new Dictionary that contains the same columns as our training dataset and assigns a random value to each.

```Python
def random_bool():
    return random_number()

def random_number(low=0, high=1):
    return random.randint(low,high)

def generate_data():
    internetServices = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    paymentMethods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']

    random_data = {
            'name':'customer',
            'Partner': random_bool(),
            'Dependents': random_bool(),
            'tenure': random_number(0,50),
            'PhoneService': random_bool(),
            'MultipleLines': random_number(-1),
            'InternetService': random.choice(internetServices),
            'OnlineSecurity': random_number(-1),
            'OnlineBackup': random_number(-1),
            'DeviceProtection': random_number(-1),
            'TechSupport': random_number(-1),
            'StreamingTV': random_number(-1),
            'StreamingMovies': random_number(-1),
            'Contract': random.choice(contracts),
            'PaperlessBilling': random_bool(),
            'PaymentMethod': random.choice(paymentMethods)
        }
    return random_data
```  
For `InternetService`, `Contract` and `PaymentMethod`, we hard code the possible values that can be used for each, and choose a random value. For the other features, if it only contained a `Yes` or `No` value in the training set, we'll assign `1` or `0`, randomly, For the features that used `Yes`, `No` and some other string, we'll use `1`, `0` and `-1`, respectively.

Next, lets go over our predict function, which is called when there is a request to the`/predict` endpoint.
```python
@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():

    random_user_data = generate_data()
    query = pd.get_dummies(pd.DataFrame(random_user_data, index=[0]))
    query = query.reindex(columns=model_columns, fill_value=0)

    #return prediction as probability in percent
    prediction = round(model.predict_proba(query)[:,1][0], 2)* 100
    return jsonify({'churn': prediction})
```

Here, we generate data for a random user, then create a Dataframe that looks just the Dataframe we used for training our model. This will have the same columns but only have 1 row of the dataset.
Finally, we'll call `predict_proba` on the model to return a vector for the likelihood of the user churning.
`[[0.79329917 0.20670083]]`
we'll take the last item in the vector, since this is the value of the user churning, round to 2 decimal places and convert to a percentage.
`prediction = round(model.predict_proba(query)[:,1][0], 2)* 100`
This will return `21.0`, which is the percentage of the user churning.
Finally, we'll return this value as JSON in a `churn` object.

Next, we'll deploy our model locally.
In your terminal, navigate to the project and run:
```bash
pip install flask pandas
```
Now, our model server is running, and we can now test it by making a GET request to the `/predict` endpoint.
![Model Server Example](images/model_server_postman.png)

The server returns a JSON response that shows the percentage of the likelihood of a random user churning.

# Web Application
In our last piece, we can now bring it all together.

In order to show our churn prediction, we'll be using a creating a [custom Conversation Event](https://developer.nexmo.com/conversation/code-snippets/event/create-custom-event) called `churn-prediction` that will be called when the model server returns its churn prediction for a given user.
In `common.js`, we'll add the following:

```javascript
function getChurnForUser(conversation) {
  //Send custom event to agent
  if (window.location.pathname == "/") {
    fetch("http://127.0.0.1:3001/predict")
    .then(response => {return response.json()})
    .then(json => {
      conversation.sendCustomEvent({ type: 'churn-prediction', body: json}).then(() => {
        console.log('custom event was sent');
      }).catch((error)=>{
        console.log('error sending the custom event', error);
      });
    })
    .catch(error => console.log('error', error));
  }
}
```

This function accepts the current conversation and calls the model server that we built previously. We've hardcoded the url and port `http://127.0.0.1:3001/predict` for ease of use. However, in a production environment, this would exist on a server.
Also, in the real world, we would pass in the user's id in order to generate a prediction for that user. But for this example, as shown before, we generate a random user's information to be used in the churn prediction model.

Next, we'll add a `h2` tag on the agent's screen to show the churn prediction, Then, we'll add a listener to `churn-prediction` event, which will update the text inside the `h2` tag.
Inside `setupListeners` function we'll add this:

```Javascript
activeConversation.on('churn-prediction', (sender, message) => {
  if (window.location.pathname == "/agent") {
    document.getElementById("churn_text").innerHTML = "Likelihood of current customer churn: " +  message["body"]["churn"] + "%"
    console.log(sender, message);
  }
});
```
When the `churn-prediction` event fires, we return send the churn prediction in the `message` property and update the `innerHTML` text with the churn.

Now, whenever we have a new user connect with our agent, we'll be able to see what's their likely of no longer needed our companies services. If the possibly of churn is high, maybe, we should be a little extra nice to them :)
