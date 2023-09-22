import torch


class DiversityLoss:
    def __init__(self, l=1):
        self.l = l

    def __resize_for_subtraction(self, element):
        left = element.unsqueeze(0).expand(element.size(0), -1, -1)
        right = element.unsqueeze(1).expand(-1, element.size(0), -1)
        return left, right

    def get_kernel(self, embedding):
        left, right = self.__resize_for_subtraction(embedding)
        return torch.exp(
            -(torch.square((left - right))).mean(-1) / (2 * self.l ** 2))

    def __call__(self, embedding):
        kernel = self.get_kernel(embedding)
        return - torch.logdet(kernel)
