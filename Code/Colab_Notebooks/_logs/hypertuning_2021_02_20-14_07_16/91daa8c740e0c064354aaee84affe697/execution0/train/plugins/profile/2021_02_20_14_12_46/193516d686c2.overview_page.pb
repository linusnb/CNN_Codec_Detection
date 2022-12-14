?	??.Q??z@??.Q??z@!??.Q??z@	?j:??t???j:??t??!?j:??t??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??.Q??z@???b@12>?^?wq@A?SH?9??IG???8@Yw/??Q???*	֣p=???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2\??AAA#@!?????X@)it?3#@1l?i-??X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?yq???!?	?FId??)?yq???1?	?FId??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchH?ξ? ??!????<??)H?ξ? ??1????<??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!R̰2?D??)???:U???1?͉?ۚ??:Preprocessing2F
Iterator::Model?M?a????!??? $??)?z6?>w?1?H?M???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?j:??t??Ip???YA@Q???|LP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???b@???b@!???b@      ??!       "	2>?^?wq@2>?^?wq@!2>?^?wq@*      ??!       2	?SH?9???SH?9??!?SH?9??:	G???8@G???8@!G???8@B      ??!       J	w/??Q???w/??Q???!w/??Q???R      ??!       Z	w/??Q???w/??Q???!w/??Q???b      ??!       JGPUY?j:??t??b qp???YA@y???|LP@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??{%/??!??{%/??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2j???!Z???<Y??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Gԅֱ?!G?"????0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputn^s??R??!?sl?v???0"=
sequential/conv_layer2/Relu_FusedConv2D?ϕ??!???e0v??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?2????!*~ѳ?e??0"=
sequential/conv_layer3/Relu_FusedConv2D=J^???!?!??????"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGradz1c?"???!U?lW??"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???eV???!޴D{?e??0"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????w??!???w m??0Q      Y@Y?|?a@@a|???P@q??,\-@y/|???!c?"?

both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?14.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 