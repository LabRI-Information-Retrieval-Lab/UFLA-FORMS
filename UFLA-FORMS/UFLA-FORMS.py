"""
UFLA-FORMS loading script for labeled documents images dataset.

Created by: Victor G. Lima (https://github.com/Victorgonl)
Original source: https://github.com/Victorgonl/UFLA-FORMS
Version: 1.1.0
License: https://github.com/Victorgonl/UFLA-FORMS/blob/main/LICENSE

For use in your dataset, follow the instructions on README.md of the original source. This script do not need modifications to load your dataset according to UFLA-FORMS. Feel free to experiment according to your needs!
"""

"""
Edit your dataset location adress on "dataset_location.py".
"""
from .dataset_location import DATASET_LOCATION

import copy
import json

import PIL.Image
import transformers
import datasets


class UFLAFORMSProcessor:
    """
    Class responsible for loading, processing and chunking the dataset.
    """

    DATA_PATH = "/data/"
    DATA_FILE_EXTENSION = "json"

    IMAGE_PATH = "/image/"
    IMAGE_FILE_EXTENSIONS = ["png", "jpg"] + [
        extension.replace(".", "") for extension in PIL.Image.registered_extensions()
    ]

    TAG_FORMATS = {"IOB2": ["B", "I"], "IOBES": ["B", "I", "E", "S"]}

    RELATION_SYMBOL = "->"
    NON_RELATION_SYMBOL = "X"

    BOX_NORMILIZER = 1000

    O_LABEL = "O"
    CLS_TOKEN_BOX = [0, 0, 0, 0]
    SEP_TOKEN_BOX = [BOX_NORMILIZER, BOX_NORMILIZER, BOX_NORMILIZER, BOX_NORMILIZER]
    PAD_TOKEN_LABEL = "O"
    PAD_TOKEN_BOX = [0, 0, 0, 0]

    def normalize_box(self, box, image_size, box_normalization):
        return [
            int(box_normalization * box[0] / image_size[0]),
            int(box_normalization * box[1] / image_size[1]),
            int(box_normalization * box[2] / image_size[0]),
            int(box_normalization * box[3] / image_size[1]),
        ]

    def normalize_boxes(self, boxes, image_size):
        return [
            self.normalize_box(box, image_size, self.BOX_NORMILIZER) for box in boxes
        ]

    def words_to_tokens_list(self, words, tokenizer: transformers.PreTrainedTokenizer):
        tokens_lists = [tokenizer.tokenize(word) for word in words]
        return tokens_lists

    def extract_words_boxes_labels_entities(
        self, image, data, tokenizer, tag_format, valid_labels
    ):
        n_tokens_entities = 0
        n_entities = 0
        input_ids = [tokenizer.cls_token_id]
        bbox = [self.CLS_TOKEN_BOX]
        labels = [self.O_LABEL]
        entities = {"start": [], "end": [], "label": []}
        entities_map = {}
        for i in range(len(data)):
            # skip empty entity
            if not data[i]["text"]:
                continue
            entity_start = len(input_ids)
            words = data[i]["words"]
            boxes = data[i]["boxes"]
            boxes = self.normalize_boxes(boxes, image.size)
            tokens_lists = self.words_to_tokens_list(words=words, tokenizer=tokenizer)
            sample_tokens, sample_boxes = [], []
            for token_list, box in zip(tokens_lists, boxes):
                for token in token_list:
                    sample_tokens.append(token)
                    sample_boxes.append(box)
            input_ids += tokenizer.convert_tokens_to_ids(sample_tokens)
            bbox += sample_boxes
            entity_end = len(input_ids)
            label = data[i]["label"]
            for j in range(len(sample_tokens)):
                if label not in valid_labels:
                    labels.append(self.O_LABEL)
                else:
                    if tag_format == "IOB2":
                        if j == 0:
                            labels.append("B-" + label)
                            n_tokens_entities += 1
                        else:
                            labels.append("I-" + label)
                    elif tag_format == "IOBES":
                        if len(sample_tokens) == 1:
                            labels.append("S-" + label)
                            n_tokens_entities += 1
                        elif j == 0:
                            labels.append("B-" + label)
                            n_tokens_entities += 1
                        elif j < len(sample_tokens) - 1:
                            labels.append("I-" + label)
                        elif j == len(sample_tokens) - 1:
                            labels.append("E-" + label)
            # if entity is a relevant
            if label in valid_labels:
                entities_map[data[i]["id"]] = len(entities["start"])
                entities["start"].append(entity_start)
                entities["end"].append(entity_end)
                entities["label"].append(label)
                n_entities += 1
        input_ids.append(tokenizer.sep_token_id)
        bbox.append(self.SEP_TOKEN_BOX)
        labels.append(self.O_LABEL)
        assert n_tokens_entities == n_entities
        return input_ids, bbox, labels, entities, entities_map

    def extract_relations(
        self, data, entities, entities_map, valid_labels, valid_relations
    ):
        relations = {"head": [], "tail": [], "label": []}
        relations_set = set()
        for i in range(len(data)):
            for link in data[i]["links"]:
                if not link:
                    continue
                if (
                    data[link[0]]["label"] not in valid_labels
                    or data[link[1]]["label"] not in valid_labels
                ):
                    continue
                if [
                    data[link[0]]["label"],
                    data[link[1]]["label"],
                ] not in valid_relations:
                    continue
                if link[0] not in entities_map or link[1] not in entities_map:
                    continue
                x, y = entities_map[link[0]], entities_map[link[1]]
                if (x, y) not in relations_set:
                    relations_set.add((x, y))
                    relations["head"].append(x)
                    relations["tail"].append(y)
                    head_label = entities["label"][x]
                    tail_label = entities["label"][y]
                    relations["label"].append(
                        f"{head_label}{self.RELATION_SYMBOL}{tail_label}"
                    )
        return relations

    def build_relation(
        self,
        sample,
        valid_relations,
        dual_state_oversampling=False,
        all_possible_relations=False,
    ):
        relations, entities = sample["relations"], sample["entities"]
        new_relations = []
        if len(entities["start"]) < 2 or len(relations["head"]) == 0:
            entities = {
                "start": [0, 0] + entities["start"],
                "end": [1, 1] + entities["end"],
                "label": [valid_relations[0][0], valid_relations[0][1]]
                + entities["label"],
            }
            new_relations = {
                "head": [0, 0, 1, 1],
                "tail": [0, 1, 0, 1],
                "label": [
                    self.NON_RELATION_SYMBOL,
                    self.NON_RELATION_SYMBOL,
                    self.NON_RELATION_SYMBOL,
                    self.NON_RELATION_SYMBOL,
                ],
            }
        else:
            positive_relations = list(
                zip(relations["head"], relations["tail"], relations["label"])
            )
            negative_relations = []
            for x, x_label in enumerate(entities["label"]):
                for y, y_label in enumerate(entities["label"]):
                    if (
                        x == y or [x_label, y_label] not in valid_relations
                    ) and not all_possible_relations:
                        continue
                    r = (x, y, f"{x_label}{self.RELATION_SYMBOL}{y_label}")
                    if r not in positive_relations:
                        r = (x, y, self.NON_RELATION_SYMBOL)
                        negative_relations.append(r)
            if dual_state_oversampling:
                negative_relations += negative_relations
                for r in positive_relations:
                    new_negative_relation = (r[0], r[1], self.NON_RELATION_SYMBOL)
                    negative_relations.append(new_negative_relation)
            reordered_relations = positive_relations + negative_relations
            new_relations = {"head": [], "tail": [], "label": []}
            new_relations["head"] = [i[0] for i in reordered_relations]
            new_relations["tail"] = [i[1] for i in reordered_relations]
            new_relations["label"] = [i[2] for i in reordered_relations]
        assert len(entities["start"]) == len(entities["end"]) == len(entities["label"])
        assert (
            len(new_relations["head"])
            == len(new_relations["tail"])
            == len(new_relations["label"])
        )
        assert len(new_relations["head"]) != 0
        assert len(entities["start"]) >= 2
        if all_possible_relations and not dual_state_oversampling:
            assert len(entities["start"]) ** 2 == len(new_relations["head"])
        sample["relations"], sample["entities"] = new_relations, entities
        return sample

    def load_sample(
        self,
        dataset_directory: str,
        sample_id: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        valid_labels,
        valid_relations,
        tag_format: str = "IOB2",
    ):

        assert tag_format in self.TAG_FORMATS

        image = None
        for image_extension in self.IMAGE_FILE_EXTENSIONS:
            try:
                image_directory = (
                    f"{dataset_directory}/image/{sample_id}.{image_extension}"
                )
                image = PIL.Image.open(image_directory)
                break
            except:
                pass
        data_directory = (
            f"{dataset_directory}/data/{sample_id}.{self.DATA_FILE_EXTENSION}"
        )
        with open(data_directory) as data_json:
            data = json.load(fp=data_json)

        input_ids, bbox, labels, entities, entities_map = (
            self.extract_words_boxes_labels_entities(
                image, data, tokenizer, tag_format, valid_labels
            )
        )
        relations = self.extract_relations(
            data, entities, entities_map, valid_labels, valid_relations
        )

        sample = {
            "id": sample_id,
            "original_id": sample_id,
            "input_ids": input_ids,
            "bbox": bbox,
            "labels": labels,
            "entities": entities,
            "relations": relations,
            "image": image,
        }
        return sample

    def encode_sample(
        self,
        sample: dict,
        tokenizer,
        image_processor,
        max_length=512,
        padding="max_length",
        pad_to_multiple_of=8,
    ) -> dict:

        assert (
            (len(sample["input_ids"]) == len(sample["bbox"]) == len(sample["labels"]))
        ) <= max_length

        sample["attention_mask"] = [1 for _ in range(len(sample["input_ids"]))]

        collated_sample = tokenizer.pad(
            sample,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        assert len(collated_sample["bbox"]) == len(collated_sample["labels"])
        for _ in range(len(collated_sample["bbox"]), max_length):
            collated_sample["bbox"].append(self.PAD_TOKEN_BOX)
            collated_sample["labels"].append(self.PAD_TOKEN_LABEL)

        # encoding image
        original_image = sample["image"].convert("RGB")
        image = image_processor(original_image, apply_ocr=False, do_resize=True)[  # type: ignore
            "pixel_values"
        ][
            0
        ]
        collated_sample["image"] = image
        collated_sample["original_image"] = original_image

        # id, entities and relations don't need encoding
        collated_sample["id"] = sample["id"]
        collated_sample["original_id"] = sample["original_id"]
        collated_sample["entities"] = sample["entities"]
        collated_sample["relations"] = sample["relations"]

        assert len(collated_sample["input_ids"]) == max_length
        assert len(collated_sample["bbox"]) == max_length
        assert len(collated_sample["labels"]) == max_length
        assert len(collated_sample["attention_mask"]) == max_length
        assert (
            len(collated_sample["entities"]["start"])
            == len(collated_sample["entities"]["end"])
            == len(collated_sample["entities"]["label"])
        )
        assert (
            len(collated_sample["relations"]["head"])
            == len(collated_sample["relations"]["tail"])
            == len(collated_sample["relations"]["label"])
        )

        return collated_sample

    def chunk_sample(
        self,
        sample: dict,
        tokenizer,
        max_length=512,
    ) -> list[dict]:
        """Creates chunks using the `sample` passed accoding to the `max_length` target.

        Args:
            sample (dict): a UFLA-FORMS-like document sample
            tokenizer: a tokenizer with support for bounding boxes.
            max_length (int, optional): max length target. Defaults to 512.

        Returns:
            list[dict]: list of samples chunks
        """

        def get_n_connected_components_with_minor_difference(connected_components, n):
            """Using a list of connect components, returns `n` lists with connect components with similar size of entities."""

            connected_components = copy.deepcopy(connected_components)

            n_connected_components = [[] for _ in range(n)]

            while (len(connected_components)) > 0:
                n_connected_components[0] += connected_components.pop(0)
                n_connected_components = sorted(n_connected_components, key=len)

            return n_connected_components

        def split_relations_using_dfs(entities: dict, relations: dict) -> list:
            adjacency_list: dict[int, list] = {}
            for key, value in zip(relations["head"], relations["tail"]):
                if key not in adjacency_list:
                    adjacency_list[key] = []
                adjacency_list[key].append(value)
                if value not in adjacency_list:
                    adjacency_list[value] = []
                adjacency_list[value].append(key)

            visited = set()
            connected_entities = []

            def dfs(node, component):
                visited.add(node)
                component.append(node)
                for neighbor in adjacency_list[node]:
                    if neighbor not in visited:
                        dfs(neighbor, component)

            for node in adjacency_list:
                if node not in visited:
                    component: list[int] = []
                    dfs(node, component)
                    connected_entities.append(component)

            for entity in range(len(entities["start"])):
                if entity not in visited:
                    connected_entities.append([entity])

            return connected_entities

        def create_sample_with_entities(
            reference_sample, tokenizer, connected_entities
        ):

            old2new = {}

            new_sample = {
                "id": None,
                "original_id": reference_sample["original_id"],
                "input_ids": [tokenizer.cls_token_id],
                "bbox": [self.CLS_TOKEN_BOX],
                "labels": [self.O_LABEL],
                "entities": {"start": [], "end": [], "label": []},
                "relations": None,
                "image": reference_sample["image"].copy(),
            }

            for entity in connected_entities:

                old2new[entity] = len(new_sample["entities"]["start"])

                start = reference_sample["entities"]["start"][entity]
                end = reference_sample["entities"]["end"][entity]
                label = reference_sample["entities"]["label"][entity]
                new_entity_start = len(new_sample["input_ids"])
                new_sample["input_ids"] += reference_sample["input_ids"][start:end]
                new_sample["bbox"] += reference_sample["bbox"][start:end]
                new_sample["labels"] += reference_sample["labels"][start:end]
                new_entity_end = len(new_sample["input_ids"])
                new_sample["entities"]["start"].append(new_entity_start)
                new_sample["entities"]["end"].append(new_entity_end)
                new_sample["entities"]["label"].append(label)

            new_sample["input_ids"].append(tokenizer.sep_token_id)
            new_sample["bbox"].append(self.SEP_TOKEN_BOX)
            new_sample["labels"].append(self.O_LABEL)

            new_relation = {"head": [], "tail": [], "label": []}
            for head, tail, label in zip(
                reference_sample["relations"]["head"],
                reference_sample["relations"]["tail"],
                reference_sample["relations"]["label"],
            ):
                if head in connected_entities and tail in connected_entities:
                    new_head = old2new[head]
                    new_tail = old2new[tail]
                    new_relation["head"].append(new_head)
                    new_relation["tail"].append(new_tail)
                    new_relation["label"].append(label)
            new_sample["relations"] = new_relation

            assert (
                len(new_sample["input_ids"])
                == len(new_sample["bbox"])
                == len(new_sample["labels"])
            )

            return new_sample

        def get_non_entities(sample, tokenizer):

            input_ids, bbox, labels = [], [], []

            indexes_of_tokens_of_entities = []
            for start, end in zip(
                sample["entities"]["start"], sample["entities"]["end"]
            ):
                indexes_of_tokens_of_entities += list(range(start, end))

            for i in range(len(sample["input_ids"])):
                input_id, box, label = (
                    sample["input_ids"][i],
                    sample["bbox"][i],
                    sample["labels"][i],
                )
                if i not in indexes_of_tokens_of_entities and input_id not in [
                    tokenizer.cls_token_id,
                    tokenizer.sep_token_id,
                ]:
                    input_ids.append(input_id)
                    bbox.append(box)
                    labels.append(label)

            assert len(input_ids) == len(bbox) == len(labels)

            return input_ids, bbox, labels

        def break_bigger_connected_entities_sets(connected_entities):
            connected_entities = sorted(connected_entities, key=len)
            bigger_entities_group = connected_entities.pop(-1)
            half = len(bigger_entities_group) // 2
            new_groups = bigger_entities_group[:half], bigger_entities_group[half:]
            connected_entities += new_groups
            return connected_entities

        def len_input_ids(sample):
            return len(sample["input_ids"])

        def create_sample(
            reference_sample, input_ids, bbox, labels, max_length, tokenizer
        ):
            new_sample = {
                "id": None,
                "original_id": reference_sample["original_id"],
                "input_ids": [tokenizer.cls_token_id],
                "bbox": [self.CLS_TOKEN_BOX],
                "labels": [self.O_LABEL],
                "entities": {"start": [], "end": [], "label": []},
                "relations": {"head": [], "tail": [], "label": []},
                "image": reference_sample["image"].copy(),
            }
            while (len(new_sample["input_ids"]) < max_length - 1) and input_ids:
                new_sample["input_ids"].append(input_ids.pop(0))
                new_sample["bbox"].append(bbox.pop(0))
                new_sample["labels"].append(labels.pop(0))
            new_sample["input_ids"].append(tokenizer.sep_token_id)
            new_sample["bbox"].append(self.SEP_TOKEN_BOX)
            new_sample["labels"].append(self.O_LABEL)

            assert len(new_sample["input_ids"]) <= max_length

            return new_sample

        # if there's no need for chunking
        if len(sample["input_ids"]) <= max_length:
            return [sample]

        original_sample = sample

        # calculate disconnected relations using DFS
        entities = original_sample["entities"]
        relations = original_sample["relations"]
        connected_entities_sets = split_relations_using_dfs(entities, relations)

        n = 1
        n_lost_relations = 0
        is_all_done = False
        n_connected_components = []

        while not is_all_done:

            if connected_entities_sets:

                n_connected_components = (
                    get_n_connected_components_with_minor_difference(
                        connected_entities_sets, n
                    )
                )
                if not all(
                    len(connected_components) > 0
                    for connected_components in n_connected_components
                ):
                    n = 2
                    connected_entities_sets = break_bigger_connected_entities_sets(
                        connected_entities_sets
                    )
                    n_lost_relations += 1
                    continue

            new_samples = []
            big_samples = []
            for connected_entities in n_connected_components:
                new_sample = create_sample_with_entities(
                    original_sample, tokenizer, connected_entities
                )
                if len(new_sample["input_ids"]) <= max_length:
                    new_samples.append(new_sample)
                else:
                    big_samples.append(new_sample)
            if big_samples:
                n += 1
                continue

            new_samples = sorted(new_samples, key=len_input_ids)

            input_ids, bbox, labels = get_non_entities(original_sample, tokenizer)

            for sample in new_samples:
                input_ids_end = sample["input_ids"].pop(-1)
                bbox_end = sample["bbox"].pop(-1)
                labels_end = sample["labels"].pop(-1)
                while (len(sample["input_ids"]) < max_length - 1) and input_ids:
                    sample["input_ids"].append(input_ids.pop(0))
                    sample["bbox"].append(bbox.pop(0))
                    sample["labels"].append(labels.pop(0))
                sample["input_ids"].append(input_ids_end)
                sample["bbox"].append(bbox_end)
                sample["labels"].append(labels_end)
                assert len(sample["input_ids"]) <= max_length

            while input_ids:
                new_sample = create_sample(
                    original_sample, input_ids, bbox, labels, max_length, tokenizer
                )
                new_samples.append(new_sample)

            new_samples = sorted(new_samples, key=len_input_ids)

            is_all_done = True

        # assign id to new samples based on the original
        for i in range(len(new_samples)):
            new_samples[i]["id"] = f"{original_sample['original_id']}_{i}"

        # sanity test
        # TODO: fix the implementation of chunk as this raise error depending on the tokenizer
        set_x: set[int] = set()
        set_y = set(original_sample["input_ids"])
        for sample in new_samples:
            assert (
                len(sample["input_ids"]) == len(sample["bbox"]) == len(sample["labels"])
            ) and len(sample["input_ids"]) <= max_length
            set_x = set_x.union(set(sample["input_ids"]))
        assert set_x == set_y
        x = 0
        for new_sample in new_samples:
            x += len(new_sample["input_ids"]) - 2
        y = len(original_sample["input_ids"]) - 2
        assert x == y

        x, w = 0, 0
        for new_sample in new_samples:
            x += len(new_sample["relations"]["head"])
            w += len(new_sample["entities"]["start"])
        y = len(original_sample["relations"]["head"])
        z = len(original_sample["entities"]["start"])
        assert w == z
        if n_lost_relations == 0:
            assert x == y - n_lost_relations

        return new_samples


class UFLAFORMSConfig(datasets.BuilderConfig):
    """Hugging Face Datasets' BuilderConfig for UFLA-FORMS like datasets, also responsible for holding additional configs."""

    def __init__(
        self,
        uflaforms_processor,
        split_option,
        labels,
        relations,
        tag_format,
        tokenizer,
        image_processor,
        max_length: int,
        original_image,
        relations_dual_state_oversampling,
        all_possible_relations,
        **kwargs,
    ):
        super(UFLAFORMSConfig, self).__init__(**kwargs)
        self.uflaforms_processor = uflaforms_processor
        assert (
            tag_format in uflaforms_processor.TAG_FORMATS
        ), f"Tag format should be one of {list(uflaforms_processor.TAG_FORMATS.keys())}"
        self.split_option = split_option
        self.labels = labels
        self.relations = relations
        self.tag_format = tag_format
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.original_image = original_image
        self.relations_dual_state_oversampling = relations_dual_state_oversampling
        self.all_possible_relations = all_possible_relations


class UFLAFORMS(datasets.GeneratorBasedBuilder):
    """
    Hugging Face Datasets' GeneratorBasedBuilder for UFLA-FORMS like datasets.
    """

    __name = "UFLA-FORMS"

    config: UFLAFORMSConfig

    default_tokenizer = "xlm-roberta-base"

    def __init__(self, *args, **kwargs):
        self.__get_dataset_info()
        self.__process_args(**kwargs)
        super().__init__(*args, **kwargs)

    def __process_args(self, **kwargs):

        split_option = kwargs["split_option"] if "split_option" in kwargs else None
        tokenizer = (
            kwargs["tokenizer"]
            if "tokenizer" in kwargs
            else transformers.AutoTokenizer.from_pretrained(self.default_tokenizer)
        )
        image_processor = (
            kwargs["image_processor"]
            if "image_processor" in kwargs
            else transformers.LayoutLMv2ImageProcessor.from_pretrained(
                "microsoft/layoutxlm-base"
            )
        )
        labels = kwargs["labels"] if "labels" in kwargs else self.dataset_info["labels"]
        relations = (
            kwargs["relations"]
            if "relations" in kwargs
            else self.dataset_info["relations"]
        )
        max_length = kwargs["max_length"] if "max_length" in kwargs else 512
        tag_format = kwargs["tag_format"] if "tag_format" in kwargs else "IOB2"
        original_image = (
            kwargs["original_image"] if "original_image" in kwargs else False
        )
        relations_dual_state_oversampling = (
            kwargs["relations_dual_state_oversampling"]
            if "relations_dual_state_oversampling" in kwargs
            else False
        )
        all_possible_relations = (
            kwargs["all_possible_relations"]
            if "all_possible_relations" in kwargs
            else False
        )

        uflaforms_processor = UFLAFORMSProcessor()

        self.BUILDER_CONFIGS = [
            UFLAFORMSConfig(
                name=self.__name,
                uflaforms_processor=uflaforms_processor,
                split_option=split_option,
                labels=labels,
                relations=relations,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_length=max_length,
                tag_format=tag_format,
                original_image=original_image,
                relations_dual_state_oversampling=relations_dual_state_oversampling,
                all_possible_relations=all_possible_relations,
            )
        ]

    def __get_dataset_info(self):

        dl_manager = datasets.download.DownloadManager()

        # TODO: implement this appropriately
        try:
            self.dataset_directory = dl_manager.extract(DATASET_LOCATION)
        except:
            try:
                self.dataset_directory = dl_manager.download_and_extract(
                    DATASET_LOCATION
                )
            except:
                self.dataset_directory = DATASET_LOCATION

        with open(f"{self.dataset_directory}/dataset_info.json") as dataset_info:
            self.dataset_info = json.load(dataset_info)

    def __ner_labels_names(self):
        labels = self.config.labels
        tags = self.config.uflaforms_processor.TAG_FORMATS[self.config.tag_format]
        labels = ["O"]
        for label in self.config.labels:
            for tag in tags:
                labels.append(f"{tag}-{label.upper()}")
        return labels

    def _info(self):
        return datasets.DatasetInfo(
            dataset_name=self.dataset_info["dataset_name"],
            description=self.dataset_info["description"],
            citation=self.dataset_info["citation"],
            homepage=self.dataset_info["homepage"],
            version=self.dataset_info["version"],
            license=self.dataset_info["license"],
            features=datasets.features.Features(
                {
                    "id": datasets.features.Value("string"),
                    "original_id": datasets.features.Value("string"),
                    "input_ids": datasets.features.Sequence(
                        datasets.features.Value("int64")
                    ),
                    "bbox": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.features.Value("int64"))
                    ),
                    "labels": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=self.__ner_labels_names(), id=self.config.tag_format
                        )
                    ),
                    "attention_mask": datasets.features.Sequence(
                        datasets.features.Value("uint8")
                    ),
                    "entities": datasets.features.Sequence(
                        {
                            "start": datasets.features.Value("int64"),
                            "end": datasets.features.Value("int64"),
                            "label": datasets.features.ClassLabel(
                                names=self.config.labels
                            ),
                        }
                    ),
                    "relations": datasets.features.Sequence(
                        {
                            "head": datasets.features.Value("int64"),
                            "tail": datasets.features.Value("int64"),
                            "label": datasets.features.ClassLabel(
                                names=[
                                    self.config.uflaforms_processor.NON_RELATION_SYMBOL
                                ]  # must be first to receive id = 0
                                + [
                                    f"{relation[0]}{self.config.uflaforms_processor.RELATION_SYMBOL}{relation[1]}"
                                    for relation in self.config.relations
                                ]
                            ),
                        }
                    ),
                    "image": datasets.features.Array3D(
                        shape=(3, 224, 224), dtype="uint8"
                    ),
                    "original_image": datasets.features.Image(),
                }
            ),
        )

    def _split_generators(self, dl_manager):

        if self.config.split_option is None:
            self.config.split_option = next(iter(self.dataset_info["splits"]))
        else:
            assert (
                self.config.split_option in self.dataset_info["splits"]
            ), f"Split option '{self.config.split_option}' not found. Split options available: {list(self.dataset_info['splits'].keys())}"
        splits = self.dataset_info["splits"][self.config.split_option]
        splits_names = list(splits.keys())

        split_generators = [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"samples": splits[split_name]},
            )
            for split_name in splits_names
        ]

        return split_generators

    def _generate_examples(self, samples):
        for sample_id in samples:

            sample = self.config.uflaforms_processor.load_sample(
                dataset_directory=self.dataset_directory,  # type: ignore
                sample_id=sample_id,
                tokenizer=self.config.tokenizer,
                valid_labels=self.config.labels,
                valid_relations=self.config.relations,
                tag_format=self.config.tag_format,
            )

            sample = {
                "id": sample["id"],
                "original_id": sample["original_id"],
                "image": sample["image"],
                "input_ids": sample["input_ids"],
                "bbox": sample["bbox"],
                "labels": sample["labels"],
                "entities": sample["entities"],
                "relations": sample["relations"],
            }

            samples = self.config.uflaforms_processor.chunk_sample(
                sample=sample,
                tokenizer=self.config.tokenizer,
                max_length=self.config.max_length,
            )

            for sample in samples:
                sample = self.config.uflaforms_processor.build_relation(
                    sample=sample,
                    valid_relations=self.config.relations,
                    dual_state_oversampling=self.config.relations_dual_state_oversampling,
                    all_possible_relations=self.config.all_possible_relations,
                )
                sample = self.config.uflaforms_processor.encode_sample(
                    sample,
                    tokenizer=self.config.tokenizer,
                    image_processor=self.config.image_processor,
                    max_length=self.config.max_length,
                )
                if not self.config.original_image:
                    del sample["original_image"]
                yield sample["id"], sample
