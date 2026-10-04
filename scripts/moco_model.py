import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MoCo25D(nn.Module):

    def __init__(
        self,
        feature_dim=128,
        queue_size=512,
        momentum=0.999,
        temperature=0.07
    ):
        super().__init__()

        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size

        # Query encoder
        self.encoder_q = resnet18(weights=None)

        self.encoder_q.conv1 = nn.Conv2d(
            5,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_features = self.encoder_q.fc.in_features

        self.encoder_q.fc = nn.Linear(
            num_features,
            feature_dim
        )

        # Momentum encoder
        self.encoder_k = resnet18(weights=None)

        self.encoder_k.conv1 = nn.Conv2d(
            5,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.encoder_k.fc = nn.Linear(
            num_features,
            feature_dim
        )

        # Start momentum encoder with same weights
        self.encoder_k.load_state_dict(
            self.encoder_q.state_dict()
        )

        # Momentum encoder is not directly trained
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue of negative features
        self.register_buffer(
            "queue",
            torch.randn(feature_dim, queue_size)
        )

        self.queue = F.normalize(
            self.queue,
            dim=0
        )

        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )

    @torch.no_grad()
    def update_momentum_encoder(self):

        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = (
                param_k.data * self.momentum
                + param_q.data * (1.0 - self.momentum)
            )

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:

            self.queue[:, ptr:ptr + batch_size] = keys.T

        else:

            first_part = self.queue_size - ptr
            second_part = batch_size - first_part

            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, :second_part] = keys[first_part:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k):

        # Query features
        q = self.encoder_q(x_q)

        q = F.normalize(
            q,
            dim=1
        )

        # Momentum features
        with torch.no_grad():

            self.update_momentum_encoder()

            k = self.encoder_k(x_k)

            k = F.normalize(
                k,
                dim=1
            )

        # Positive similarity
        positive = torch.sum(
            q * k,
            dim=1,
            keepdim=True
        )

        # Negative similarity
        negative = torch.mm(
            q,
            self.queue.clone().detach()
        )

        # Combine positive and negative logits
        logits = torch.cat(
            [positive, negative],
            dim=1
        )

        logits /= self.temperature

        # Positive example is class 0
        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device
        )

        # Add current keys to queue
        self.dequeue_and_enqueue(k)

        return logits, labels