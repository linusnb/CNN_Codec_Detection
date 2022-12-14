?	??C}%?@??C}%?@!??C}%?@	 ?K?8??? ?K?8???! ?K?8???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??C}%?@???cj?e@1K?4?{@A??b? ̹?I?N??@Y8-x?W???*	5^?ILD?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2G?Z?Q? @!j????X@)?'??] @1?}i4?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?Qԙ{H??!??ċ'q??)?Qԙ{H??1??ċ'q??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch0o????!e	Z????)0o????1e	Z????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?2?g??!??;&???)?sE)!X??1[??m??:Preprocessing2F
Iterator::ModelC?????!?K???t??)\Y???"t?1?Ue??8??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 28.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?K?8???I\????=@Q6?7??Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???cj?e@???cj?e@!???cj?e@      ??!       "	K?4?{@K?4?{@!K?4?{@*      ??!       2	??b? ̹???b? ̹?!??b? ̹?:	?N??@?N??@!?N??@B      ??!       J	8-x?W???8-x?W???!8-x?W???R      ??!       Z	8-x?W???8-x?W???!8-x?W???b      ??!       JGPUY ?K?8???b q\????=@y6?7??Q@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!???????0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputf'9???!?s?w???0"=
sequential/conv_layer2/Relu_FusedConv2D?!t????!`?
-????"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ۜ0???!F?1????0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?h??????!WO?????0"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInput?i?Y7???!G?"????0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?/gJ??!Þ?Tk???0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGradx?{???!D?5c???"=
sequential/conv_layer3/Relu_FusedConv2Dv~Bt?ң?!+?yF?1??"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh??:??!~k:?\+??0Q      Y@Y??D??D<@a??.???Q@q???4@y?ů8?Y?"?

both?Your program is POTENTIALLY input-bound because 28.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 