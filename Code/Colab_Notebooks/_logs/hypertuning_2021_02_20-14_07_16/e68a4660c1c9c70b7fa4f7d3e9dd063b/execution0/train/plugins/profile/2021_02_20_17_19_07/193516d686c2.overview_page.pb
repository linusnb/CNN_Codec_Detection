?	iSu??!?@iSu??!?@!iSu??!?@	#?P\ȵ?#?P\ȵ?!#?P\ȵ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6iSu??!?@?????c@1?3?ތ?@A??!?????I?O?Y?#@Y?}?e????*	F?z???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2oG8-x#@!?i?yJ?X@)?????"@1??O??~X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@/m8,???!????G??)/m8,???1????G??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??l??!?`a?/???)4??E`???12??O????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?B?_?+??!?????)?B?_?+??1?????:Preprocessing2F
Iterator::Model?M?»??!??e?a???)??g??t?1?)?`?F??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9"?P\ȵ?I??Q?w3@Q?iWp4T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????c@?????c@!?????c@      ??!       "	?3?ތ?@?3?ތ?@!?3?ތ?@*      ??!       2	??!???????!?????!??!?????:	?O?Y?#@?O?Y?#@!?O?Y?#@B      ??!       J	?}?e?????}?e????!?}?e????R      ??!       Z	?}?e?????}?e????!?}?e????b      ??!       JGPUY"?P\ȵ?b q??Q?w3@y?iWp4T@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterێ???5??!ێ???5??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP? ?*T??!???D{???0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputD` ??X??!??X???0"=
sequential/conv_layer2/Relu_FusedConv2D?}?C???!??.????"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterޏhpd>??!?P5?????0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?C?W???!?t?=(@??0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad??a ??!???M.R??"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^E?????!???B?'??0"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGrad??߁?K??!/??fO???"=
sequential/conv_layer3/Relu_FusedConv2D???k᱙?!dL!r????Q      Y@Y{?[R?j;@a?i+S%R@q?c=BBm.@y2??P?"?

both?Your program is POTENTIALLY input-bound because 18.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 