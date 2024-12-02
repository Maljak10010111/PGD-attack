from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
import torch
import open_clip
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from ingredients.utilities import resize_cifar10_img


class ClipModel():

    def __init__(self, model_name, processor_name, tokenizer_name, use_open_clip, label_names, torch_preprocess,
                 dataset, device):

        self.instantiate_model(model_name, processor_name, tokenizer_name, use_open_clip, torch_preprocess)
        self.device = device
        self.model.to(self.device)
        self.dataset = dataset
        self.labels = label_names
        if dataset == "cifar10":
            self.clip_labels = [f"this is a photo of a {label}" for label in self.labels]
        else:
            self.clip_labels = [f"photo of  a {label}" for label in self.labels]
        self.instantiate_label_embeddings()

    def instantiate_model(self, model_name, processor_name, tokenizer_name, use_open_clip, torch_preprocess=None):
        if use_open_clip:
            model, _, processor = open_clip.create_model_and_transforms(model_name)
            model.eval()
            tokenizer = open_clip.get_tokenizer(tokenizer_name)
        else:
            processor = CLIPProcessor.from_pretrained(processor_name)
            vision_model = CLIPVisionModel.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(processor_name)
            model.vision_model.load_state_dict(vision_model.vision_model.state_dict())
            tokenizer = None

        self.use_open_clip = use_open_clip
        self.torch_processor = torch_preprocess
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        model.eval()

    def instantiate_label_embeddings(self):
        if self.use_open_clip:
            text = self.tokenizer(self.clip_labels)
            text_features = self.model.encode_text(text.to(self.device))
            text_features = F.normalize(text_features, dim=-1)
            self.label_emb = text_features
        else:
            self.label_tokens = self.processor(
                text=self.clip_labels,
                padding=True,
                images=None,
                return_tensors='pt'
            ).to(self.device)

            self.label_emb = self.model.get_text_features(**self.label_tokens)
            self.label_emb = self.label_emb.to(self.device)
            self.label_emb /= torch.norm(self.label_emb, dim=0)

    def cosine_similarity(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def create_img_emb(self, batch_image):
        if self.use_open_clip:
            if self.torch_processor is not None:
                processed_images = torch.stack([self.torch_processor(img) for img in batch_image])
            else:
                to_pil = ToPILImage()
                processed_images = torch.stack([self.processor(to_pil(img)) for img in batch_image])

            image = processed_images.to(self.device)
            image_features = self.model.encode_image(image)
            img_emb = F.normalize(image_features, dim=-1)
        else:
            processed_images = torch.stack([self.torch_processor(img) for img in batch_image])
            if self.dataset == "cifar10":
                resized_img = resize_cifar10_img(processed_images)
            else:
                resized_img = processed_images
            img_emb = self.model.get_image_features(resized_img)
            img_emb = img_emb.to(self.device)
        return img_emb

    def clip_prediction(self, img_emb, labels):
        labels = labels.to(self.device)

        scores = torch.mm(img_emb, self.label_emb.transpose(0, 1))

        preds = torch.argmax(scores, dim=1)

        correct_predictions = (preds == labels).float()

        return correct_predictions


class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super(ClipClassifier, self).__init__()
        self.clip = clip_model

    def parameters(self):
        return self.clip.model.parameters()

    def eval(self):
        self.clip.model.eval()

    def forward(self, x):
        image_encoding = self.clip.create_img_emb(x)
        return torch.abs(self.clip.cosine_similarity(image_encoding, self.clip.label_emb))
