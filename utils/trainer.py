import torch


# def train(net, criterion, optimizer, trainloader, devloader, device="cpu", epoch_n=100, path="./checkpoint/save.pt"):
def train(net, optimizer, data_loader, device, epoch_n):
    for epoch in range(epoch_n): # loop over the dataset multiple times
        net.train()
        results = [0,0,0,0]
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            loss_dict = net(images, targets)

            # 'loss_classifier', 'loss_box_reg', loss_mask', 'loss_objectness', 'loss_rpn_box_reg'
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            results = [(a+float(b.item())) for a,b in zip(results, loss_dict.values())]
            print(i, ' ', [f"{i.item():.4f}" for i in loss_dict.values()])
            # print(torch.cuda.memory_snapshot())
            # print(torch.cuda.memory_stats())
            exit()

        print(f'[{epoch + 1:3}/{epoch_n}] loss: {losses}')

        # net.eval()
        # for data in data_loader:
        #     img, target = data
        #     output = net(list(i.to(device).detach() for i in img))[0]
        #     break

        # print(output["boxes"].shape)
        # print(output["masks"].shape)
        # print(torch.unique(output["masks"][0]))

        # log result
        with open("log.csv", "a") as f:
            if epoch == 0:
                mask_loss_temp = results[2]
                f.write(f"loss_classifier, loss_box_reg, loss_mask, loss_objectness\n")
            else:
                if results[2] < mask_loss_temp:
                    torch.save(net.state_dict(), "best_model.pth")
                    mask_loss_temp = results[2]

            for result in results[:-1]:
                f.write(f"{result:.6f},")
            f.write(f"{results[-1]:.6f}\n")
         
        ## Save model
        torch.save(net.state_dict(), "latest_model.pth")


        # validate the model 

        # net.eval()
        # for i, data in enumerate(devloader, 0):
        #     # move tensors to GPU if CUDA is available
        #     inputs, labels, data_index = data
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)

        #     # forward pass: compute predicted outputs by passing inputs to the model
        #     outputs = net(inputs)

        #     # calculate the batch loss
        #     loss = criterion(outputs, labels)
        #     # update average validation loss 
        #     valid_loss += loss.item()*inputs.shape[0]

        # # calculate average losses
        # train_loss = train_loss/len(trainloader.dataset)
        # valid_loss = valid_loss/len(devloader.dataset)
    
        # # print training/validation statistics 
        # print(f'Training Loss: {train_loss:.6f} Validation Loss: {valid_loss:.6f}')
        
    print('Finished Training')

   
