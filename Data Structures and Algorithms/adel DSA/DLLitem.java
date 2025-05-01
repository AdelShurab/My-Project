public class DLLitem {
    static class DLLItems {
        Nodeitem.NodeItem head, tail;

        public void addItem(item.Item item) {
            Nodeitem.NodeItem newItem = new Nodeitem.NodeItem(item);
            if(head == null) {
                head = tail = newItem;
            } else {
                tail.next = newItem;
                newItem.prev = tail;
                tail = newItem;
            }
        }

    }
}
