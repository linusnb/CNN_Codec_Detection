?	˃???@˃???@!˃???@	??m??ʸ???m??ʸ?!??m??ʸ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6˃???@c??U?d@14??u@AD0.s??I͓k
dV@YHlw?}??*	<?O?w??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2eM.?P$@!?GW??X@)=E7'$@18?*???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??j??Ǵ?!$:?q?e??)??j??Ǵ?1$:?q?e??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Z&????!?~8j{???)??Z&????1?~8j{???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??bFx{??!????L%??)I0??Z
??1?Ц?<b??:Preprocessing2F
Iterator::Model?̕A???!?)?xԤ??)臭???s?1RdcQx???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 32.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??m??ʸ?I6?Pi?@@Q?f???P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c??U?d@c??U?d@!c??U?d@      ??!       "	4??u@4??u@!4??u@*      ??!       2	D0.s??D0.s??!D0.s??:	͓k
dV@͓k
dV@!͓k
dV@B      ??!       J	Hlw?}??Hlw?}??!Hlw?}??R      ??!       Z	Hlw?}??Hlw?}??!Hlw?}??b      ??!       JGPUY??m??ʸ?b q6?Pi?@@y?f???P@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterM? K??!M? K??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ؐ???!????|??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput Qcu??!?W?g??0"=
sequential/conv_layer2/Relu_FusedConv2D???D???!??%o???"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter䕽vS???!???nC???0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInputF??,?F??!?,˱????0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?O ?????!?1]?Vg??0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad??S???!{lRظ??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z?B???!$Rq7\??0"=
sequential/conv_layer3/Relu_FusedConv2Da??z???!???].??Q      Y@Y?Z??Z??@aC)?B)Q@q2v?z5@y*rM??C]?"?

both?Your program is POTENTIALLY input-bound because 32.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?21.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 