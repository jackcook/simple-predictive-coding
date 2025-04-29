import click


@click.command()
@click.option("--activations-learning-rate", type=float, default=1e-2)
@click.option("--batch-size", type=int, default=128)
@click.option("--dataset", type=click.Choice(["mnist", "cifar10"]), required=True)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
)
@click.option("--model", type=click.Choice(["mlp", "vgg5"]), required=True)
@click.option("--num-epochs", type=int, default=10)
@click.option("--num-relaxation-steps", type=int, default=8)
@click.option("--weights-learning-rate", type=float, default=1e-5)
def train(
    activations_learning_rate: float,
    batch_size: int,
    dataset: str,
    device: str,
    model: str,
    num_epochs: int,
    num_relaxation_steps: int,
    weights_learning_rate: float,
):
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from utils import get_dataloaders, get_model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size)
    model = get_model(model).to(device)

    # Two optimizers are used in predictive coding: This first one optimizes
    # the model parameters, while the second one optimizes the model
    # activations.
    parameters_optimizer = torch.optim.AdamW(
        model.parameters(), lr=weights_learning_rate
    )

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} of {num_epochs}")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            # Disable gradient computation for the model parameters: they're
            # not needed in the first step of predictive coding.
            for var in model.parameters():
                var.detach_()

            activations = model(x)
            y_pred = activations[-1]

            # In predictive coding, the first activation is fixed to the input,
            # and the last activation is fixed to the correct output. This
            # allows us to optimize the series of activations that lie between
            # the input and the output.
            activations[-1] = nn.functional.one_hot(y, num_classes).float()

            for i in range(len(model.layers)):
                if isinstance(model.layers[i], nn.Flatten):
                    # Flatten layers don't have learnable parameters like
                    # Linear or Conv2d layers, so they don't need to be
                    # optimized and we can skip them here.
                    continue

                # Set requires_grad to True so that gradients are computed for
                # each activation tensor.
                activations[i].requires_grad_()

            # This optimizer optimizes the activations of each layer. Note the
            # first parameter: only the activations are optimized here, not the
            # model parameters.
            activations_optimizer = torch.optim.SGD(
                activations, lr=activations_learning_rate
            )

            # Perform several relaxation steps, through which we try to minimize
            # each layer's prediction error by finding better activations.
            for i in range(num_relaxation_steps):
                activations_optimizer.zero_grad()

                # Compute the mean squared error of each layer's prediction
                # error. Remember that activations_optimizer is updating the
                # activations, so each activation tensor changes in each step,
                # but the model parameters are kept constant.
                loss = sum(
                    0.5
                    * torch.sum(
                        (activations[i] - model.layers[i - 1](activations[i - 1])) ** 2
                    )
                    for i in range(1, len(activations))
                )
                loss.backward()

                activations_optimizer.step()

            # After the relaxation steps, use the new optimized activations to
            # optimize the model parameters.
            parameters_optimizer.zero_grad()

            # Enable gradient computation for the model parameters, reversing
            # the detaching we did earlier.
            for var in model.parameters():
                var.requires_grad_()

            # Compute the exact same mean squared error loss as before, but now
            # optimize the model parameters while keeping the activations
            # constant.
            loss = sum(
                0.5
                * torch.sum(
                    (activations[i] - model.layers[i - 1](activations[i - 1])) ** 2
                )
                for i in range(1, len(activations))
            )
            loss.backward()

            parameters_optimizer.step()

        # Calculate accuracy on the test set at the end of each epoch.
        test_acc = []

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            activations = model(x)
            y_pred = activations[-1]
            acc = (y_pred.argmax(dim=-1) == y).float().mean()
            test_acc.append(acc)

        print(f"Train loss: {loss.item():.4f}")
        print(f"Test accuracy: {torch.tensor(test_acc).mean().item():.4f}")
        print()


if __name__ == "__main__":
    train()
