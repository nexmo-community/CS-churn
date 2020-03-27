let activeConversation;
let application;
let didSendChurnPrediction = false

function setupConversation(apiPath) {
  fetch(apiPath) /* To generate the JWT for the agent */
    .then(function(response) {
      console.log("setupConversation",response)
      return response.json();
    })
    .then(function(response) {
      new NexmoClient({
        debug: false
      })
        .login(response.jwt) /* Used to log into Nexmo */
        .then(app => {
          application = app
          console.log('*** Logged into app', app);
          return app.getConversation(response.conversation.id); /* Grabs conversation from Nexmo's server */
        })
        .then(conversation => {
          console.log('*** Retrieved conversations', conversation);
          activeConversation = conversation;
          setupListeners();
        })
        .catch(console.error);
    });
}

function setupListeners() {
  console.log("***** setupListeners ****")
  const form = document.getElementById('textentry');
  const textbox = document.getElementById('textbox');
  activeConversation.on('text', (sender, message) => {
    console.log(sender, message);
    appendMessage(
      message.body.text,
      `${sender.user.name === 'agent' ? 'agent' : 'input'}`
    );
  });

  activeConversation.on('churn-prediction', (sender, message) => {
    if (window.location.pathname == "/agent") {
      console.log("message", message)
      document.getElementById("churn_text").innerHTML = "Likely hood of current customer churn: " +  message["body"]["churn"] + "%"
      // alert(message)
      console.log(sender, message);
    }
  });

  form.addEventListener('submit', event => {
    console.log("event", event)
    console.log("****")
    event.preventDefault();
    event.stopPropagation();
    const inputText = textbox.value;

    activeConversation.sendText(inputText);
    if (!didSendChurnPrediction) {
      getChurnForUser(activeConversation)
      didSendChurnPrediction = true
    }

    textbox.value = '';
  }, false);
}

let messageId = 0;

function getChurnForUser(conversation) {
  //Send custom event to agent
  if (window.location.pathname == "/") {
    fetch("http://127.0.0.1:3001/predict",{
    mode: 'cors',
    headers: {
      'Access-Control-Allow-Origin':'*'
    }
  })
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

function appendMessage(message, sender, appendAfter) {
  const messageDiv = document.createElement('div');
  messageDiv.classList = `message ${sender}`;
  messageDiv.innerHTML = '<p>' + message + '</p>';
  messageDiv.dataset.messageId = messageId++;

  const messageArea = document.getElementById('message-area');
  if (appendAfter == null) {
    messageArea.appendChild(messageDiv);
  } else {
    const inputMsg = document.querySelector(
      `.message[data-message-id='${appendAfter}']`
    );
    inputMsg.parentNode.insertBefore(messageDiv, inputMsg.nextElementSibling);
  }

  messageArea.scroll({ /* Scroll the message area to the bottom. */
    top: messageArea.scrollHeight,
    behavior: 'smooth'
  });

  return messageDiv.dataset.messageId; /* Return this message id so that a reply can be posted to it later */
}
