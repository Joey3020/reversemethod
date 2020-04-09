def train(net, model_name, patience):
    
    optimizer = optim.Adam(net.parameters(), LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss_function = nn.MSELoss()
    
    validate_every = 2

    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    with open("model.log", "a") as f:
        for epoch in tqdm(range(EPOCHS)):
                net.train()
                for idx in range(traindata.getbatchnum()):
                    x, y = traindata.getbatch()
                    x = x.view(-1, 1, 64, 64)

                    net.zero_grad()
                    pred = net(x)
                    loss = loss_function(pred, y)
                    loss.backward()
                    optimizer.step()
                    
                    error = (pred - y) / y
                    error = error.mean(0)
                    train_losses.append(loss.item())
                    
                    #if idx%validate_every == 0:
                net.eval()
                for ii in range(valdata.getbatchnum()):
                    val_x, val_y = valdata.getbatch()
                    val_x = val_x.view(-1, 1, 64, 64)
                    val_pred = net(val_x)
                    val_loss = loss_function(val_pred, val_y)
                    val_error = (val_pred - val_y) / val_y
                    val_error = val_error.mean(0)
                    val_losses.append(val_loss.item())
                    
                train_loss = np.average(train_losses)
                valid_loss = np.average(val_losses)
                avg_train_losses.append(train_loss)
                avg_val_losses.append(valid_loss)
                
                train_losses = []
                val_losses = []
                
                f.write(f"{MODEL_NAME},{round(time.time(),3)},  {round(float(loss), 4)},  {round(float(val_loss),4)}\n")
                print("batch : ", idx)
                print("loss : ", loss, "\nval loss : ", val_loss, "\n")
                print("error : ", error, "\nval error : ", val_error)
                
                early_stopping(valid_loss, net)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        # load the last checkpoint with the best model
        net.load_state_dict(torch.load('checkpoint.pt'))
        
        return avg_train_losses, avg_val_losses

def test(net):
    errors = []
    predictions = []
    net.eval()
    with torch.no_grad():
        for idx in range(testdata.getbatchnum()):
            data, label = testdata.getbatch()
            pred = net(data)

            error = (label - pred) / label
            #print(error)
            predictions.append(pred.numpy())
            errors.append(error.numpy())
    return predictions, errors
