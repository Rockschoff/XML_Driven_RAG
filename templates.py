from langchain_core.prompts import ChatPromptTemplate

updatePersistent="""
Persistent Memory if the memory of the all the important things that you should keep in mind. It contains\\
information about the characters that your intertacing with, what they are doing  and all the important information\\
you should remeber about a person. You should also use the Persistent memory to store tasks that your asked to remember\\
or that you think you should perform. Persistent memory should be stored in a compact XML type format where each tag has a time attribute\\
that shows the time the tag was last updated or added. Eg of a persistent memry element is as follows:\\

<Person:Ron time='02/01/2024 12:05:00hrs'> is going to buy food <Task:Grocery-Reminder  time='02/01/2024 12:07:00hrs>Remind him to buy milk and egs</Task:Task:Grocery-Reminder></Person:Ron>.\\

You are given the new User-Message (with current timestamp) , Recent-Chat-History and the current Persistent-Memory. Based on the conversation \\
you have to update the persistent memory wherever necessary. Only store the information about the state of objects and that
is informative, not need to store everything about the conversation. **Only give the XML for the Persistent-Memory and nothing else**.



User-Message ({timestamp}) : {user_message}
Recent-Chat-History        : {recent_chat_history}
Persistent-Memory          : {persistent_memory}
Updated-Persistent-Memory  :"""

getPersistentContext="""
Persistent Memory if the memory of the all the important things that you should keep in mind. It contains\\
information about the characters that your intertacing with, what they are doing  and all the important information\\
you should remeber about a person. You should also use the Persistent memory to store tasks that your asked to remember\\
or that you think you should perform. Persistent memory should be stored in a compact XML type format where each tag has a time attribute\\
that shows the time the tag was last updated or added. Eg of a persistent memry element is as follows:\\

<Person:Ron time='02/01/2024 12:05:00hrs'> is going to buy food <Task:Grocery-Reminder  time='02/01/2024 12:07:00hrs>Remind him to buy milk and egs</Task:Task:Grocery-Reminder></Person:Ron>.\\

You are given the new User-Message (with current timestamp) , Recent-Chat-History and the updated Persistent-Memory. Based on the conversation \\
and the Persistent-Memory (that is updated) you have to extract information that can be helpful  and provide context you your response . Be sure to include any tasks that are due at this timestamp or event\\

Give only the context and noting else and keep it short and informative

User-Message ({timestamp}) : {user_message}
Recent-Chat-History        : {recent_chat_history}
Updated-Persistent-Memory  : {persistent_memory}
Context  :"""

getFinalAnswer="""

You are a wonderful AI assisstant use this information to give answer to the User-Input. Here Persistent-Memory is the information about the state of the
things that you are talking about, Hazy-Memory are similar conversaations that you have had with user in the past
and Chat-History is the most recent chat till now.

Your response should be helpful and in confirmation with the Memory, but no need to reference thing from the 
memory if it is not needed. Give only your response and nothing else.

Persistent-Memory : {persistent_memory}

Hazy-Memory : {hazy_memory}

Chat-History : {chat_history}

User-Input   : {user_input}

timestamp : {timestamp}

AI Assistant :
"""

getCompressed="""

you are given a XML file that shows the Persistent Memory of an AI Assisstant. The AI Assisstant is running out of memory space and need to
compress the Persistent Memory. Persistent Memory is the memory about all the tasks, people, places , objects the AI Assistant should remember.
 Each XML tag has a timestamp of when it was last updated. Compression should be such that oldest tags should be compressed the most.
While compressing remember that the goal is to remember the current state of objects, people and places to the best your ability.

**Respond by giving the XML text only and nothing else.**

Persistant-Memory : {persistent_memory}

Compressed-Persistant-Memory : 

"""

updatePersistentTemplate = ChatPromptTemplate.from_template(updatePersistent)
getPersistentContextTemplate=ChatPromptTemplate.from_template(getPersistentContext)
getFinalAnswerTemplate=ChatPromptTemplate.from_template(getFinalAnswer)
getCompressedTemplate = ChatPromptTemplate.from_template(getCompressed)