?	T?^pB?@T?^pB?@!T?^pB?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-T?^pB?@???c?>f@1m??]??t@A??CR%??I????η@*	'1?I?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??S?@!?????X@)?t??@1f????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@Xp?????!-?Z<????)Xp?????1-?Z<????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch	?%qVD??!>?"?$??)	?%qVD??1>?"?$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^gE?D??!hO@??4??)?j??? ??1?"??* ??:Preprocessing2F
Iterator::ModelI?<?+J??!ޮ?t???)?1??|z?1??2+?Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIN?^?A@Q?X??PP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???c?>f@???c?>f@!???c?>f@      ??!       "	m??]??t@m??]??t@!m??]??t@*      ??!       2	??CR%????CR%??!??CR%??:	????η@????η@!????η@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qN?^?A@y?X??PP@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,P?-ݛ??!,P?-ݛ??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltercrt{???!ȱQ,???0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?j;*???!??%Զb??0"=
sequential/conv_layer2/Relu_FusedConv2Df?ձ???!Z??l???"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??[L????!?E????0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput/?ЧB???!?N??C5??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterѿ??0???!??????0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad+?W???!陟?????"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInputb?ݳ톤?!?s?uW9??0"=
sequential/conv_layer3/Relu_FusedConv2DQ???p??!??85e`??Q      Y@Y?Z??Z??@aC)?B)Q@q??y+bXB@yT???&b?"?

both?Your program is POTENTIALLY input-bound because 34.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?36.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 