import os
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import get_image, preproc

Image.MAX_IMAGE_PIXELS = None  # remove DecompressionBombWarning


class MyData(data.Dataset):
    def __init__(self, config, dataset_dir, image_size, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.preproc_methods = config.preproc_methods
        self.keep_size = not config.img_size
        self.data_size = [config.img_size, config.img_size]
        self.is_train = is_train
        self.load_all = config.load_all

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.data_size[::-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize(self.data_size[::-1]),
                transforms.ToTensor(),
            ]
        )
        self.image_paths = []

        # 修改 1：指向 Imgs 文件夹，并兼容 jpg 和 png
        image_root = os.path.join(dataset_dir, "Imgs")
        self.image_paths += [
            os.path.join(image_root, p)
            for p in os.listdir(image_root)
            if p.endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG"))
        ]

        self.label_paths = []
        for p in self.image_paths:
            base, ext = os.path.splitext(p)  # 分离文件名和扩展名
            p_gt = os.path.join(
                dataset_dir, "GT",
                os.path.basename(base) + ".png",
            )
            # === 修改点 1：如果标签不存在，用 None 占位，绝对保证列表长度一致 ===
            if os.path.exists(p_gt):
                self.label_paths.append(p_gt)
            else:
                self.label_paths.append(None)

        self.edge_paths = []
        for p in self.image_paths:
            base, ext = os.path.splitext(p)  # 分离文件名和扩展名
            p_edge = os.path.join(
                dataset_dir, "Edge",
                os.path.basename(base) + ".png",
            )
            # === 修改点 2：如果边缘不存在，用 None 占位，绝对保证列表长度一致 ===
            if os.path.exists(p_edge):
                self.edge_paths.append(p_edge)
            else:
                self.edge_paths.append(None)

        # === 修改点 3：只在训练时进行严格断言，测试时缺胳膊少腿没关系 ===
        if self.is_train and None in self.label_paths:
            raise ValueError(f"训练集缺少 GT 标签！图片数: {len(self.image_paths)}")
        if self.is_train and None in self.edge_paths:
            raise ValueError(f"训练集缺少 Edge 标签！图片数: {len(self.image_paths)}")

        if self.load_all:
            self.images_loaded, self.labels_loaded, self.edges_loaded = [], [], [] 
            # === 修改点 4：打包时把 edge_paths 也放进去 ===
            for i, (image_path, label_path, edge_path) in enumerate(
                    tqdm(
                        zip(self.image_paths, self.label_paths, self.edge_paths),
                        total=len(self.image_paths),
                    )
            ):
                _image = get_image(image_path, size=self.data_size, color_type="rgb")
                
                # === 修改点 5：有路径就读取，是 None 就生成全黑假图（Dummy Image） ===
                _label = get_image(label_path, size=self.data_size, color_type="gray") if label_path else Image.new('L', tuple(self.data_size), 0)
                _edge = get_image(edge_path, size=self.data_size, color_type="gray") if edge_path else Image.new('L', tuple(self.data_size), 0)

                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.edges_loaded.append(_edge)

    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            edge = self.edges_loaded[index] 
        else:
            image = get_image(
                self.image_paths[index], size=self.data_size, color_type="rgb"
            )
            
            # === 修改点 6：__getitem__ 中同理，防范 None 引发越界和类型错误 ===
            if self.label_paths[index] is not None:
                label = get_image(self.label_paths[index], size=self.data_size, color_type="gray")
            else:
                label = Image.new('L', tuple(self.data_size), 0)
                
            if self.edge_paths[index] is not None:
                edge = get_image(self.edge_paths[index], size=self.data_size, color_type="gray")
            else:
                edge = Image.new('L', tuple(self.data_size), 0)

        # 预处理（只在训练时做数据增强）
        if self.is_train:
            image, label, edge = preproc(
                image,
                label,
                edge,
                preproc_methods=self.preproc_methods,
            )

        # 转换为 Tensor
        image = self.transform_image(image)
        label = self.transform_label(label)
        edge = self.transform_label(edge) # 无论训练还是验证，都要转 Tensor

        return image, label, edge

    def __len__(self):
        return len(self.image_paths)
