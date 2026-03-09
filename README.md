# multimodal-topic-segmentation-hgt
Multimodal Topic Segmentation in Lecture Videos Using Heterogeneous Graph Transformers


The rapid growth of online education platforms has led to a significant increase in the availability of lecture videos and instructional materials. However, these lecture recordings are often lengthy and lack explicit topic boundaries, making it difficult for learners to efficiently navigate and retrieve relevant information. Traditional topic segmentation approaches primarily rely on textual transcripts and ignore the multimodal nature of instructional content, which frequently includes mathematical equations, tables, diagrams, and other visual elements. This limitation reduces the effectiveness of segmentation methods when applied to complex lecture material.

This research proposes a multimodal topic segmentation framework for lecture videos based on Heterogeneous Graph Transformers (HGT). The proposed approach models lecture content as a sequence of instructional units, where each unit may correspond to different modalities such as textual explanations, mathematical expressions, tables, or diagrams. By integrating information from multiple modalities, the framework captures richer semantic relationships between lecture components and improves the accuracy of topic boundary detection.

The pipeline begins with an Instructional Unit Builder, which decomposes lecture content into smaller semantic units. Each unit is then assigned a modality label, such as text, equation, table, or diagram, and normalized through a Heterogeneous Lecture Content (HLC) representation. To generate meaningful feature representations, modality-specific encoders are applied. These include language models for textual content, mathematical encoders for equations, and vision-based encoders for diagrams and tables. The embeddings produced by these encoders are projected into a shared semantic space using a cross-modal projection layer.

After obtaining multimodal embeddings, the instructional units are represented as nodes in a heterogeneous graph structure, where edges capture relationships between units across time and modality. A Heterogeneous Graph Transformer is then applied to learn contextual interactions between nodes, allowing the model to capture both sequential lecture flow and cross-modal dependencies. This graph-based representation enables the framework to identify structural patterns within lecture content that correspond to topic transitions.

Topic boundaries are detected using a change-point detection mechanism that analyzes similarity shifts between adjacent instructional units. To further improve segmentation efficiency, FAISS-based semantic similarity search is used to measure embedding proximity across the lecture timeline. This combination of graph representation learning and similarity profiling allows the system to accurately detect topic transitions in heterogeneous lecture content.

The proposed framework is evaluated using a collection of publicly available academic datasets along with a synthetic multimodal lecture dataset designed to simulate realistic instructional scenarios. These datasets provide different modality combinations, including text-only documents, scientific articles containing mathematical expressions, and document layouts with tables and diagrams. The synthetic dataset extends these resources by incorporating fully multimodal instructional sequences.

Experimental results demonstrate that the proposed multimodal segmentation framework effectively captures cross-modal relationships and improves topic boundary detection compared with traditional text-based approaches. By modeling lecture content through heterogeneous graph structures, the system provides a scalable and flexible solution for segmenting complex educational material.

Overall, this work contributes a novel multimodal topic segmentation architecture, a unified dataset representation for heterogeneous instructional content, and a reproducible experimental pipeline for analyzing lecture materials. The proposed approach has potential applications in educational video navigation, automated lecture summarization, and intelligent learning systems.
Dataset Preparation

Since the proposed multimodal topic segmentation pipeline integrates multiple datasets originating from different sources and possessing heterogeneous structures, all datasets must first be converted into a unified representation before they can be processed by the segmentation framework. Each dataset originally follows its own structural format depending on the domain, annotation style, and dataset design. Therefore, a preprocessing and conversion stage is required to normalize these heterogeneous inputs into a consistent schema compatible with the proposed pipeline. This unified representation enables the system to seamlessly process instructional units from different datasets while preserving modality information and temporal ordering.

The unified dataset format required by the pipeline is stored in a tabular CSV structure containing five fields: doc_id, unit_id, temporal_index, content, and modality. The doc_id represents the unique identifier of the lecture or document from which the instructional unit originates. The unit_id specifies the identifier of each instructional unit within the document. The temporal_index represents the chronological position of the instructional unit within the lecture or document timeline, ensuring that the sequential structure of the content is preserved. The content field stores the textual representation of the instructional unit, which may include textual explanations, mathematical expressions, table descriptions, or figure captions. The modality field indicates the modality type associated with the content and can take one of the following values: text, equation, table, or diagram.

This standardized format allows all stages of the proposed segmentation pipeline to operate consistently across datasets. Specifically, the unified representation supports the Instructional Unit Builder, which organizes lecture content into discrete units, the HLC normalization process, which ensures consistent modality labeling, the modality-specific encoders used for multimodal representation learning, and the cross-modal projection layer responsible for aligning embeddings from different modalities. Furthermore, the unified dataset format enables the construction of heterogeneous graphs used by the Heterogeneous Graph Transformer (HGT), supports change-point detection for identifying topic boundaries, and allows efficient similarity search through FAISS-based semantic segmentation.

The experimental evaluation utilizes several publicly available academic datasets, each contributing different instructional modalities commonly found in lecture materials and scientific documents. The LectureBank dataset contains lecture transcripts that are structured using lecture identifiers, section titles, and paragraph-level text segments. During conversion, the lecture identifier is mapped to the unified doc_id, paragraph text is assigned to the content field, and the paragraph order within the lecture defines the temporal_index. Since LectureBank primarily consists of textual lecture transcripts, the modality for these instructional units is labeled as text. When mathematical expressions or figures are present, simple pattern-based rules are applied to detect equation-like expressions or diagram references.

The arXiv Math/CS dataset, obtained from the publicly available Kaggle repository, contains scientific articles that include structured sections, textual content, equations, figures, and tables. In the conversion process, the paper_id is mapped to doc_id, and each section of the article is decomposed into instructional units following the order in which they appear in the document. The textual components of the article are stored as text modality units, while mathematical expressions are labeled as equation modality units. Table descriptions are assigned the table modality, and figure captions are labeled as diagram modality units. The section ordering within the document determines the temporal_index, thereby preserving the logical flow of the scientific article.

The DocBank dataset is a document layout analysis dataset that provides fine-grained annotations for document tokens along with bounding box coordinates and semantic labels. In the conversion process, tokens labeled as paragraph text are grouped into coherent paragraph-level instructional units. These units are then mapped to the text modality in the unified format. Tokens annotated as equations are assigned the equation modality, while tokens corresponding to tables or figure captions are mapped to table and diagram modalities respectively. The sequential order of tokens and paragraphs in the original document is used to generate the temporal_index.

The S2ORC dataset is a large-scale scientific corpus that provides structured JSON representations of research articles, including metadata, sections, paragraphs, equations, figures, and tables. During dataset conversion, the paper_id is used as the doc_id, and each section of the article is decomposed into instructional units. Section text is mapped to the text modality, equations are assigned the equation modality, table structures are mapped to the table modality, and figures or figure captions are labeled as diagram modality units. The ordering of sections and paragraphs in the JSON structure determines the temporal_index, ensuring that the semantic flow of the document is maintained.

The GROTOAP2 dataset contains annotated scientific articles that include structured blocks such as paragraphs, equations, tables, and figures along with layout coordinates. In the conversion process, each annotated block is treated as an instructional unit. Blocks labeled as paragraphs are mapped to the text modality, blocks labeled as equations are assigned the equation modality, and blocks representing tables or figures are mapped to table and diagram modalities respectively. The order in which these blocks appear within the document determines the temporal_index used by the unified dataset format.

In addition to the publicly available datasets, a synthetic heterogeneous lecture content (HLC) dataset is generated to simulate multimodal lecture scenarios. This dataset is designed to represent typical instructional sequences encountered in lecture videos, where textual explanations are frequently interleaved with mathematical equations, tables, and diagrams. Each synthetic lecture contains a sequence of instructional units with associated modality labels and timestamps. Since the synthetic dataset is generated directly in the required structure, only minimal formatting adjustments are necessary to align it with the unified CSV schema used by the pipeline.

Once all datasets have been converted into the standardized format, they are passed to the dataset conversion pipeline. This preprocessing stage transforms the raw dataset structures into the unified representation required by the segmentation framework. The resulting dataset can then be directly consumed by the segmentation pipeline, enabling consistent multimodal processing across heterogeneous lecture content sources.










 






