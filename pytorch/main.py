import torch

from LSTMNetwork import LSTMNetwork
from QuadraticFunction import QuadraticFunction

torch.autograd.set_detect_anomaly(True)

def main():
    optimizer_network = LSTMNetwork()
    optimizer_optimizer = torch.optim.Adam(optimizer_network.parameters())

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)

        hidden = None

        quadratic_function = QuadraticFunction(10)
        theta = torch.rand(10, requires_grad=True)

        optimizer_loss = torch.zeros(1)

        for step in range(50):
            loss = quadratic_function(theta)

            loss.backward(retain_graph=True)

            grads = theta.grad.detach()

            optimizer_output, hidden = optimizer_network(grads, hidden)
            
            hidden = ([each.data for each in hidden])

            theta = theta.add(optimizer_output)

            theta = theta.requires_grad_()

            optimizer_loss = optimizer_loss + loss

            if step % 10 == 0:
                print("  Loss: ", loss.data)
        
        # optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))
        break

if __name__ == "__main__":
    main()
