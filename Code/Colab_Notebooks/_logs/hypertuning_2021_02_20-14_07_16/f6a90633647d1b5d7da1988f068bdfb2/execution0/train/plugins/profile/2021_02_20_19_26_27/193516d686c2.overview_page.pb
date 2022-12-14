?	W??Y??@W??Y??@!W??Y??@	Z??x4??Z??x4??!Z??x4??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6W??Y??@S?'?ݳd@1??9\H?@A?˛õ??I~8H???@Yd> Й4??*	9??v?&?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?]??q"@!?/?0??X@)?P29?K"@1 !????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?7k???!*ABy??)?7k???1*ABy??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch|??&??!@?1?????)|??&??1@?1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism˻????!1??????){?ю~??1Cf?d$???:Preprocessing2F
Iterator::Modele???V??!?.??????)?D?e??v?10?f????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Z??x4??I??ߵ||$@Q??&?)mV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S?'?ݳd@S?'?ݳd@!S?'?ݳd@      ??!       "	??9\H?@??9\H?@!??9\H?@*      ??!       2	?˛õ???˛õ??!?˛õ??:	~8H???@~8H???@!~8H???@B      ??!       J	d> Й4??d> Й4??!d> Й4??R      ??!       Z	d> Й4??d> Й4??!d> Й4??b      ??!       JGPUYZ??x4??b q??ߵ||$@y??&?)mV@?"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?!?5x???!?!?5x???0"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?<0M???!F/\?E"??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ٿ??!??D???0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput??)Ré?!h-?fN]??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?*vMR\??!??????0"=
sequential/conv_layer2/Relu_FusedConv2D8p?j???!W7?Qt??"=
sequential/conv_layer3/Relu_FusedConv2Do??@??!o(R?e???"=
sequential/conv_layer4/Relu_FusedConv2D???*ɤ?!?f?0??"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInputu??M??!???q8a??0"\
;gradient_tape/sequential/maxpool_layer2/MaxPool/MaxPoolGradMaxPoolGrad<???dq??!w!????Q      Y@Y?}G??1@a? ??>?T@q?\???3,@y
>Ӡ?6?"?

both?Your program is POTENTIALLY input-bound because 10.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?14.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 