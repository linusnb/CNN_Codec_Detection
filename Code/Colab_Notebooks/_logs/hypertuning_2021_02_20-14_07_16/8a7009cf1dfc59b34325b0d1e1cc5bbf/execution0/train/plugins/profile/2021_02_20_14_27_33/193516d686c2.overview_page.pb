?	?U,~?6p@?U,~?6p@!?U,~?6p@	??n?
?????n?
???!??n?
???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?U,~?6p@??FL?b@1A*Ŏ?8Z@AUm7?7M??I}w+K$@Yy??????*	?/?4?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2Hp#e?\!@!??i??X@)? L?9!@1y????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@e?VAt??!=?3?m???)e?VAt??1=?3?m???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch؝?<????!??G???)؝?<????1??G???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Z
H???!?_$?޳??)?/L?
F??1s?CW??:Preprocessing2F
Iterator::Model???q?j??!0rˋ??)?A
?B?t?1????~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??n?
???I????M@Q/e??e7D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??FL?b@??FL?b@!??FL?b@      ??!       "	A*Ŏ?8Z@A*Ŏ?8Z@!A*Ŏ?8Z@*      ??!       2	Um7?7M??Um7?7M??!Um7?7M??:	}w+K$@}w+K$@!}w+K$@B      ??!       J	y??????y??????!y??????R      ??!       Z	y??????y??????!y??????b      ??!       JGPUY??n?
???b q????M@y/e??e7D@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!???????0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInputP;??????!T?}.;???0"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1?????!??@?H??0"=
sequential/conv_layer2/Relu_FusedConv2D?H??[??!?@?ʟ??"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?^?~#!??!???????0"-
IteratorGetNext/_1_Send?E??ͽ??!???~Ë??"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh?]@??!?|?Z?O??0"k
Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?????8??!s??YM??"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad$"??~??!?ɫ?:???"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGrad9??>r??!9ď?[???Q      Y@Y????[E@a?tk~X?L@q^?t Y%@y??N??]u?"?

both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 