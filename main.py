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
) -> None:
    import torch
    from torch import nn
    from tqdm import tqdm

    from utils import get_dataloaders, get_model

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        )

    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size)
    model = get_model(model).to(device)

    # Keep track of the layers that have learnable parameters. Layers without learnable
    # parameters, such as Flatten or ReLU, can't be optimized, so we will skip them in
    # some later steps.
    learnable_layer_indices = {
        i
        for i, layer in enumerate(model.layers)
        if any(p.requires_grad for p in layer.parameters())
    }

    # Two optimizers are used in predictive coding: This first one optimizes the model
    # parameters, while the second one optimizes the model activations.
    parameters_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=weights_learning_rate,
    )

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)  # noqa: PLW2901

            # Disable gradient computation for the model parameters: they're not needed
            # in the first step of predictive coding.
            for var in model.parameters():
                var.detach_()

            activations = model(x)

            # In predictive coding, the first activation is fixed to the input, and the
            # last activation is fixed to the correct output. We can then optimize the
            # activations of the layers in between by reducing each layer's prediction
            # error.
            activations[-1] = nn.functional.one_hot(y, num_classes).float()

            for i in range(len(model.layers)):
                if i not in learnable_layer_indices:
                    continue

                # Set requires_grad to True so that gradients are computed for each
                # activation tensor.
                activations[i].requires_grad_()

            # This optimizer optimizes the activations of each layer. Note the first
            # parameter: only the intermediate activations are optimized here, not the
            # input, output, or any model parameters.
            activations_optimizer = torch.optim.SGD(
                activations[1:-1],
                lr=activations_learning_rate,
            )

            # Perform several relaxation steps, through which we try to minimize each
            # layer's prediction error by finding better activations.
            for _ in range(num_relaxation_steps):
                activations_optimizer.zero_grad()

                # Compute the mean squared error of each layer's prediction error.
                # Remember that activations_optimizer is updating the activations, so
                # each activation tensor changes in each step, but the model parameters
                # are kept constant.
                loss = sum(
                    torch.sum(
                        (activations[i + 1] - model.layers[i](activations[i])) ** 2,
                    )
                    for i in range(1, len(activations) - 1)
                    if i in learnable_layer_indices
                )
                loss.backward()

                activations_optimizer.step()

            # After the relaxation steps, use the new optimized activations to optimize
            # the model parameters.
            parameters_optimizer.zero_grad()

            # Enable gradient computation for the model parameters, reversing the
            # detaching we did earlier.
            for var in model.parameters():
                var.requires_grad_()

            # Compute the exact same mean squared error loss as before, but now
            # optimize the model parameters while keeping the activations constant.
            loss = sum(
                torch.sum((activations[i + 1] - model.layers[i](activations[i])) ** 2)
                for i in range(len(model.layers))
                if i in learnable_layer_indices
            )
            loss.backward()

            parameters_optimizer.step()

        # Calculate accuracy on the test set at the end of each epoch.
        test_acc = []

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # noqa: PLW2901
            activations = model(x)
            y_pred = activations[-1]
            acc = (y_pred.argmax(dim=-1) == y).float().mean()
            test_acc.append(acc)

        print(f"Train loss: {loss.item():.4f}")
        print(f"Test accuracy: {torch.tensor(test_acc).mean().item():.4f}")
        print()


if __name__ == "__main__":
    train()
