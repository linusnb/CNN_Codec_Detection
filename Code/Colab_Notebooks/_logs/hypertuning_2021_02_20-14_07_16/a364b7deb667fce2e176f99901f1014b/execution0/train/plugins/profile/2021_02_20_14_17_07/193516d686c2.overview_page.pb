?	!>???D?@!>???D?@!!>???D?@	?\?9?ݰ??\?9?ݰ?!?\?9?ݰ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6!>???D?@?e????b@1???n?@A?D?+g??Ie???-@Ym?M??*	Q??n???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?Hg`?"@!|@?G?X@)??2??!@1??l?5?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??\????!?
??	??)??\????1?
??	??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchLݕ]0???!?]???:??)Lݕ]0???1?]???:??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism|)<hvݳ?!V??
?N??)?u?r???1??+,b??:Preprocessing2F
Iterator::Model???B????!Q??????) ֪]r?1?O N?ب?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?\?9?ݰ?IL?mW??5@Q?V??S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?e????b@?e????b@!?e????b@      ??!       "	???n?@???n?@!???n?@*      ??!       2	?D?+g???D?+g??!?D?+g??:	e???-@e???-@!e???-@B      ??!       J	m?M??m?M??!m?M??R      ??!       Z	m?M??m?M??!m?M??b      ??!       JGPUY?\?9?ݰ?b qL?mW??5@y?V??S@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???4e???!???4e???0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ޏ?H??!?/?ؓN??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput̸ZE/???!?????0"=
sequential/conv_layer2/Relu_FusedConv2Du??ﭵ?!??ε͜??"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad?=ߍ????!p??????"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInputw?a?P??!gز?;??0"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteru??????!X?th;??0"=
sequential/conv_layer3/Relu_FusedConv2D????˺??!???3W??"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGradS".G>???!zK?a??"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput????M??!?3?H,??0Q      Y@YZT??5?<@a?????Q@qMDY?p,@yX͏?^\S?"?

both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?14.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 