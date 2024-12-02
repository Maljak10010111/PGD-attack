import torch

class MultiShield(torch.nn.Module):
    def __init__(self, dnn, clip_model, dataset, tolerance = 0.001):
        super(MultiShield, self).__init__()
        self.dnn = dnn
        self.clip = clip_model
        self.dataset = dataset
        self.tolerance = tolerance


    def forward(self, inputs):

        dnn_raw_predictions = self.dnn(inputs)

        dnn_predicted_labels = dnn_raw_predictions.argmax(dim=-1)

        image_encoding = self.clip.create_img_emb(inputs)

        cosine_similarity = self.clip.cosine_similarity(image_encoding, self.clip.label_emb)

        cos_sim_max, _ = torch.max(cosine_similarity, dim=1)

        cosine_i = cosine_similarity[torch.arange(cosine_similarity.size(0)), dnn_predicted_labels]

        rejection_score = torch.max(dnn_raw_predictions, dim=1)[0] + torch.abs(cos_sim_max - cosine_i) - self.tolerance

        return torch.cat((dnn_raw_predictions, rejection_score.unsqueeze(1)), dim=1)
