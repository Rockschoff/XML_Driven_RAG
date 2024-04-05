from chains import getFinalAnswer_chain, save_list_to_vectorstore, history

print("Type 'exit()' to quit.")
while True:
    
    user_input=input(">>>> ")
    if user_input.strip()=="exit()":
        save_list_to_vectorstore(history)
        break
    else:
        ans = getFinalAnswer_chain.invoke(user_input)
        print(">>>> AI:  " + ans)