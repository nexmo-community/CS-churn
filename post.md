Nexmo’s Conversation API allows developers to build their own Contact Center solution. With voice, text and integrations to other solutions, Nexmo allows you to build a complex solution, without the development being complex.

When building your own contact center solution, you need it to be smart. The most powerful solutions use AI to route calls, translate text, recommend products and so on.  Whats great, is that you don’t have to be a AI researcher with a PHD to integrate AI into your application. You also don’t have to rely on a 3rd party system, if you want to have everything in house. This seems impossible or unlikely at first, but there has been lots of progress on many machine learning libraries so that you as a developer and build a machine learning system into your solution.

In this post, we’ll look into adding a way to predict the likelihood of a customer churning. Churn, by definition is the number of customers who stopped using your product, divided by the number of total customers. For example, a company with a 1% churn per month with 1000 customers, means that 10 out of 1000 customers stop using the company's service. This can be measured by month, quarter, year and so on. It depends as a company how this is tracked. Overall, it is a great

A contact center is one place where customers interact directly with the business, especially with customer service. And losing a customer with bad support could have a large impact on churn and therefore, your health as a company.

We’ve built a simple demo application that simulates a conversation between a customer and agent, using the Conversation Service SDK.

{gif of demo}

# Overview
In our demo, we have 2 user personas, a customer and an agent. And our demo allows these users to chat, For this example, we’ll assume the company is a TV service provider, and the customer has a question about billing. We also assume that this customer has been a customer for a good amount of time, and we have data to support this.
In our example, we have some data about the user.  This would be how long in months that the customer has used the service, what their billing issue, what services they have and so on.

For our demo, when the customer interacts with the agent, we show the likelihood of the user churning on the agent’s screen, before they begin to interact. This could be helpful to the agent before starting the conversation, so that more attention could be given to the customer, depending on the likelihood of churn

# Prerequisites
This tutorial expands on a recent blog post that shows how to build an on-page live chat app. We will add to the code that can be remixed on Glitch here. You can also build the application locally using Ngrok.
 If you are not familiar with Ngrok, please refer to our Ngrok tutorial before proceeding.

Before going over the application, we first need to build our model.
The code to build the notebook is located on google colab. For this tutorial, we will assume you have a basic understanding of what machine learning is, but you wont need to fully understand everything in order to follow along.

In order to build a model, we first need data. And for this example, we’ll use Telecom Churn Dataset from IBM. This dataset contents 7043 rows of a telcom’s user data. This data is only used for examples and does not reflect a real company's users. First, let's have a look with the first 10 rows of the data using Pandas. Pandas is a python library that is used for processing and understanding data. For each user, we have 23 columns, also known as features. These include the customer gender(male, female), tenure(how long have they been a customer) and if  they have different services, including phone, internet and tv.
In order to build our model, we first need to make sure there no empty values in the dataset. If we don't check for this and try to build our model, we will have errors.
Let’s read in our dataset and remove any empty values
```python
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.dropna(axis='columns', inplace=True)
```
Next, we can use `df.head()` to view the first 10 rows of the data. When we look at the first 10 rows of data, we see that many of the rows contain strings. (YES, NO, No phone service etc) . We now have to convert these strings into numbers, because our machine learning model only knows how to deal with numbers.
For each row that contains a string, we need to see if the strings are unique. Have a look at Partner. To view all the possible values in this column, pandas has a function, called unique() to do this for us.
```python
df.Partner.unique()
array(['Yes', 'No'], dtype=object)
```

This means that the row only contains the values YES and NO. For this column, we can convert those strings into booleans(1,0)
However, if we look at the other rows, say `PaymentMethod` There’s more values than YES or NO

```python
df.MultipleLines.unique()
array(['No phone service', 'No', 'Yes'], dtype=object)
```

So for this column, we need to do a little more work. What we can do is that whenever a value is YES or NO, we can convert it to 1 or 0, respectively. When its any other string, let's set it to `-1`. Again, the machine learning model can only use numbers, so that’s why we set it to -1. It can be any number you like, but I think -1 makes sense.
If we look at the other columns, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies rows all are similar. So let's write a very simple function that goes through each column and converts our strings into ints.

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

Numeric_features is a list of all the columns that we need to update. `to_numeric` is a function that takes in the value from every row and converts the string to an int. Finally, we’ll loop through all the items in `to_numeric` and call the pandas function `apply` to call our function.
Let’s have a look at the first 10 rows to verify

{screenshot of dataframe}
It looks like those rows are now valid, but we have to deal with the other columns. Lets first inspect 'Contact' and see what values are shown
```python
df.Contract.unique()
```
which returns
`array(['Month-to-month', 'One year', 'Two year'], dtype=object)`

These values are still strings, but it's not as easy as converting to 1's and 0's. There are other columns in this dataset that are similar, including 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'

Luckily, in pandas, we can convert these values using `get_dummies()`. This function, when applied to a column, will create a new column for every possible value. And every value in each of these new columns will be `1` or `0`. This is also known as one-hot-encoding.
So for example, let’s take the `Contract` column, which contains the values of 'Month-to-month', 'One year', 'Two year'. using get_dummies(), we will create 3 new columns called `Contract_Month-to-month`, `Contract_One year` and `Contract_Two year`. And every value in these columns will be either 1 or 0.

```python
categorical_features = [
 'Partner',
 'Dependents',
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
First, we'll create a list of these features that we want to convert(`categorical_features`) and called `get_dummies` using the dataFrame(`df`) and the list of columns('categorical_features').
Lets again view the first 10 rows to double check our work.

{Sceenshot}

Looks like all the columns are numeric. Let's build our model.
To build our model, we'll use another package called scikit learn.
First, we need to 2 matrixes, `X` and `y`. `X` is a matrix that includes all of our features except for the feature that we are using to make predictions, 'Churn'. `Y` is just the value of `Churn`, which is a `1` or `0`.
```python
X = df.drop(labels='Churn',axis=1)
Y = df.Churn
print(X.shape, Y.shape)
```
this returns
`((7043, 15), (7043,))`
Which means that there are 7043 rows and 15 columns in X. For Y, we have 7043 rows.
Next, we have 2 split our data into a train and test set. When training our model, we only use a portion of the dataset. The rest of the data is used for testing. This is see how well how our model has learned the data. Generally, a testing set contains about 70% of the data. Using scikit-learn, lets split our dataset using `train-test-split` function
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)
```
Finally, we can build a model.
For this example, we will use a simple model. But in the real world, we need to test different model with our dataset to see whats best. This is where alot of the work lies in machine learning. There are no hard or fast rules to selecting a model for your data. It always depends on what the dataset.
For this example, we will use [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to train our model. Its a good model to start out with since it generally works well with making predictions for boolean values. For our application, did the user churn(1) vs did the user stay with the company(0).
Scikit learn makes it very simple to implement this model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = LogisticRegression()
lr = model.fit(X_train, y_train)
```
Once the model is fit, which also means that the model was trained on the training data, we can now test it, using the predict function on the test data
```python
model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))
````
using `metrics.accuracy_score` we can print out our accuracy, 0.8135352579271179 which is ~89%. This is means that when our model is given data to make a prediction, its result is correct about 89% percent of the time. Note, this accuracy is just one of many metrics when [evaulating a model](https://scikit-learn.org/stable/modules/model_evaluation.html).

Once we have our model, we'll need to save it in order to be used in our contact center app. To save we use the [`joblib` function dump()](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html) to save our model. Lets now build a simple Python server to serve this model.
Another thing to mention is that we also need to save the columns that we used for training. This will be used when we build our server to make predictions.
```Python
model_columns = list(df.columns)
model_columns.remove('Churn')
joblib.dump(model_columns, 'model_columns.pkl')
```

# Model Serving using Flask
Next, we will use Flask to create a basic server application to host our model. We need a way to host and model and make predictions. If this was a production application, we would need to save user information, which would contain the same information that we used for traning in our dataset. These would include how long the user has been with the company(tenure), weather they have InternetService, PhoneService, OnlineBackup etc...
This way, when we make our prediction, we would make a query to our database to return the users info, make a prediction from our model, and return the likeyhood of churning. However, for this example, we will make up information for a user and send that prediction back to the application.

In Server.py, we will make a simple Flask app, which only has one endpoint `/predict`. This endpoint will simulate users's data, invoke our model and make the prediction. Then, it will return the preciction as a JSON response
Before making the prediction, we first need to load the saved model when the server starts
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
Here we use `joblib` to load our model and the columns that we used for training. In the `predict()` function, we'll geneate a random user, create a new DataFrame from the returned data, using the saved coloumns names, to create the dataFrame
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
For InternetService, Contract and PaymentMethod, we hardcode the possible values that can used for each, and choose a random value. For the other features, if it only contained a Yes or No value in the traning set, we'll assign 1 or 0, randomly, For the features that used Yes, No and some other string, we'll use 1, 0 and -1, respectively.

Lets n 
