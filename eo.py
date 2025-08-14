import click


@click.command()
@click.option("--batch-size", type=int, default=128)
@click.option("--dataset", type=click.Choice(["mnist", "cifar10"]), required=True)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
)
@click.option("--errors-learning-rate", type=float, default=1e-3)
@click.option("--model", type=click.Choice(["mlp", "vgg5"]), required=True)
@click.option("--num-epochs", type=int, default=10)
@click.option("--num-relaxation-steps", type=int, default=8)
@click.option("--weights-learning-rate", type=float, default=5e-4)
def train(
    batch_size: int,
    dataset: str,
    device: str,
    errors_learning_rate: float,
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
    # parameters, while the second one optimizes the model activations. Note that Adam,
    # not AdamW, is typically used for optimizing parameters in error optimization.
    parameters_optimizer = torch.optim.Adam(
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
            y_pred = activations[-1]
            y_true = nn.functional.one_hot(y, num_classes).float()

            # In error optimization, prediction errors are updated during relaxation
            # rather than activations. We initialize them to zero here, and we will
            # optimize them in a moment. Note that we don't optimize the prediction
            # error of the output layer: this is handled separately below.
            errors = [
                torch.zeros_like(
                    activations[layer_i + 1],
                    requires_grad=layer_i in learnable_layer_indices,
                )
                for layer_i in range(len(model.layers) - 1)
            ]

            # This optimizer optimizes each layer's prediction error.
            errors_optimizer = torch.optim.SGD(
                [e for e in errors if e.requires_grad],
                lr=errors_learning_rate,
            )

            # Perform several relaxation steps, through which we try to minimize each
            # layer's prediction error. Remember that the model parameters are kept
            # constant during this process.
            for _ in range(num_relaxation_steps):
                errors_optimizer.zero_grad()

                # Compute the mean squared error of each layer's prediction error.
                loss = 0.5 * sum(
                    torch.sum(errors[i] ** 2)
                    for i in range(len(errors))
                    if errors[i].requires_grad
                )

                # Perform a forward pass through the model, using the current
                # prediction errors.
                s_i = x
                for e_i, layer_i in zip([*errors, 0.0], model.layers, strict=True):
                    s_i = e_i + layer_i(s_i)

                # Add the mean squared error of the output layer's prediction error.
                loss += 0.5 * torch.sum((s_i - y_true) ** 2)
                loss.backward()

                errors_optimizer.step()

            # After the relaxation steps, use the new optimized prediction errors to
            # update the model parameters.
            parameters_optimizer.zero_grad()

            # Enable gradient computation for the model parameters, reversing the
            # detaching we did earlier.
            for var in model.parameters():
                var.requires_grad_()

            # Do another forward pass through the model using the optimized prediction
            # errors, but now optimize the model parameters while keeping the
            # prediction errors constant.
            loss = torch.tensor(0.0, device=device)

            s_i = x
            for i in range(len(model.layers)):
                s_i_pred = model.layers[i](s_i)

                if i == len(model.layers) - 1:
                    # For the last layer, compare the prediction with the true label.
                    loss_i = 0.5 * torch.sum((s_i_pred - y_true) ** 2)
                else:
                    s_i = (errors[i] + s_i_pred).detach()
                    loss_i = 0.5 * torch.sum((s_i_pred - s_i) ** 2)

                if i not in learnable_layer_indices:
                    continue

                loss_i.backward(inputs=model.layers[i].parameters())
                loss += loss_i

            parameters_optimizer.step()

        # Calculate accuracy on the test set at the end of each epoch.
        test_acc = []

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # noqa: PLW2901
            y_pred = model(x)[-1]
            acc = (y_pred.argmax(dim=-1) == y).float().mean()
            test_acc.append(acc)

        print(f"Train loss: {loss.item():.4f}")
        print(f"Test accuracy: {torch.tensor(test_acc).mean().item():.4f}")
        print()


if __name__ == "__main__":
    train()
